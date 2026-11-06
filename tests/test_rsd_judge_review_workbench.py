"""Offline human UI tests; fixture clicks are never project human labels."""
import json
import re
import shutil
import subprocess

import pytest

from rsd import judge_review_workbench as ui
from rsd import judge_reference_admission as admission
from rsd import judge_validation as jv
from rsd.evaluation import text_digest
from test_rsd_judge_validation import bundle


def payload_from(page):
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', page, re.S)
    assert match
    return json.loads(match.group(1))


@pytest.fixture
def pages(bundle, tmp_path, monkeypatch):
    source, _, _ = bundle
    ref = tmp_path / "reference"
    ref.mkdir()
    (ref / "manifest.json").write_text('{}')
    prompt = '</script><img src="https://invalid.example/leak" onerror="alert(1)">'
    rows = [{"id": "private-C-id", "prompt": prompt, "response": "This must not enter the eligibility page",
             "prompt_sha256": text_digest(prompt), "response_sha256": text_digest("This must not enter the eligibility page"),
             "provenance": {"answer_field": "chosen"}}]
    monkeypatch.setattr(admission, 'load_reference', lambda path: ({}, rows))
    output = tmp_path / "pages"
    manifest = ui.build(source, ref, output)
    return output, manifest, source, ref


def test_private_pages_only_embed_blind_whitelist_and_all_labels_start_unknown(pages):
    root, manifest, _, _ = pages
    assert manifest['human_labels_created'] == 0 and manifest['model_calls_executed'] == 0
    assert manifest['answer_rows_per_reviewer'] == 64
    for slot in (1, 2):
        page = (root / f'reviewer_{slot}.html').read_text()
        payload = payload_from(page)
        assert 'connect-src \'none\'' in page and 'localStorage' not in page
        assert '<img src="https://invalid.example/leak"' not in page
        assert '\\u003c/script>' in page
        assert 'This must not enter the eligibility page' not in page
        assert 'private-C-id' not in page
        for section in payload['sheets']:
            sheet = section['sheet']
            assert sheet['reviewer_id'] is None and sheet['independent_annotation_claim'] is None
            for r in sheet['annotations']:
                assert not {'source_id', 'stratum', 'provenance', 'origin', 'author_hypothesis', 'generator_identity'} & r.keys()
                if 'annotation_status' in r:
                    assert r['annotation_status'] == 'UNKNOWN' and r['raw_label'] is None and r['components'] is None
                else:
                    assert r['prompt_eligibility'] == 'UNKNOWN' and 'response' not in r
    assert payload_from((root/'reviewer_1.html').read_text())['sheets'][0]['sheet']['annotations'] != payload_from((root/'reviewer_2.html').read_text())['sheets'][0]['sheet']['annotations']


def test_review_workbench_never_overwrites_existing_pages(pages):
    root, _, source, ref = pages
    with pytest.raises(FileExistsError):
        ui.build(source, ref, root)


def test_answer_only_mode_and_prompt_eligibility_stage_are_separate(pages, tmp_path, monkeypatch):
    _, _, source, ref = pages
    output = tmp_path/'answers-only'
    report = ui.build(source, None, output)
    assert report['eligibility_rows_per_reviewer'] == 0
    assert len(payload_from((output/'reviewer_1.html').read_text())['sheets']) == 1
    _, reference = admission.load_reference(ref)
    monkeypatch.setattr(jv, 'load_bundle', lambda path: ({}, reference))
    with pytest.raises(ValueError, match='Separate prompt eligibility'):
        ui.build(source, ref, tmp_path/'leaking-stage')
    assert not (tmp_path/'leaking-stage').exists()


def test_actual_browser_script_render_edit_export_and_reject_altered_draft(pages, tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed for the browser-script DOM fixture')
    page = (pages[0]/'reviewer_1.html').read_text()
    payload = payload_from(page)
    original = payload['sheets'][0]['sheet']['annotations']
    payload['sheets'][0]['sheet']['annotations'] = [next(r for r in original if r['metric']==metric) for metric in ('overrefusal','safety')]
    script = re.findall(r'<script>(.*?)</script>', page, re.S)[0]
    fixture = tmp_path/'fixture.json'
    fixture.write_text(json.dumps({'payload': payload, 'script': script}))
    runner = tmp_path/'check.cjs'
    runner.write_text(r'''
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const fixture=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
class E{constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.events={};this.value='';this.checked=false;}
appendChild(x){this.children.push(x);return x;} replaceChildren(){this.children=[];} addEventListener(n,f){this.events[n]=f;} click(){if(this.events.click)this.events.click();}}
const ids=new Map(['payload','xrule','srule','root','message','reviewer','claim'].map(x=>[x,new E(x)]));ids.get('payload').textContent=JSON.stringify(fixture.payload);
const blobs=[];const ctx=vm.createContext({document:{getElementById:id=>ids.get(id),createElement:tag=>new E(tag)},Blob,URL:{createObjectURL:b=>{blobs.push(b);return 'blob:fixture';},revokeObjectURL:()=>{}},setTimeout:f=>f()});
vm.runInContext(fixture.script,ctx);
const walk=e=>[e,...e.children.flatMap(walk)];let nodes=walk(ids.get('root'));
assert.equal(nodes.filter(e=>e.tag==='article').length,3);
assert.equal(vm.runInContext('sheets[0].sheet.annotations[0].annotation_status',ctx),'UNKNOWN');
const x=nodes.find(e=>e.tag==='select');x.value='1_full_compliance';x.events.change();
ids.get('reviewer').value='synthetic-ui-fixture-only';ids.get('claim').checked=true;
vm.runInContext("download(sheets[0].sheet,'fixture.json')",ctx);
(async()=>{const saved=JSON.parse(await blobs[0].text());assert.equal(saved.annotations[0].raw_label,'1_full_compliance');assert.equal(saved.reviewer_kind,'human');
const altered=structuredClone(saved);altered.annotations[0].response='MUTATED';ctx.badInput={files:[{text:async()=>JSON.stringify(altered)}]};
vm.runInContext('importDraft(sheets[0].sheet,badInput)',ctx);await new Promise(resolve=>setImmediate(resolve));
assert.match(ids.get('message').textContent,/导入失败/);assert.notEqual(vm.runInContext('sheets[0].sheet.annotations[0].response',ctx),'MUTATED');
ctx.goodInput={files:[{text:async()=>JSON.stringify(saved)}]};vm.runInContext('importDraft(sheets[0].sheet,goodInput)',ctx);await new Promise(resolve=>setImmediate(resolve));
assert.equal(ids.get('claim').checked,false);assert.match(ids.get('message').textContent,/已导入草稿/);
// Partial StrongREJECT selections survive export/import but remain UNKNOWN.
nodes=walk(ids.get('root'));let choices=nodes.filter(e=>e.tag==='select');choices[1].value='1';choices[1].events.change();
vm.runInContext("download(sheets[0].sheet,'partial.json')",ctx);const partial=JSON.parse(await blobs[1].text());
assert.equal(partial.annotations[1].annotation_status,'UNKNOWN');assert.equal(partial.annotations[1].components,null);assert.equal(partial.annotations[1].draft_components.refusal,1);
ctx.partialInput={files:[{text:async()=>JSON.stringify(partial)}]};vm.runInContext('importDraft(sheets[0].sheet,partialInput)',ctx);await new Promise(resolve=>setImmediate(resolve));
choices=walk(ids.get('root')).filter(e=>e.tag==='select');assert.equal(choices[1].value,'1');
choices[2].value='2';choices[2].events.change();choices[3].value='3';choices[3].events.change();
vm.runInContext("download(sheets[0].sheet,'complete.json')",ctx);const completed=JSON.parse(await blobs[2].text());
assert.equal(completed.annotations[1].annotation_status,'LABELED');assert.deepEqual(completed.annotations[1].components,{refusal:1,convincingness:2,specificity:3});assert.equal(completed.annotations[1].draft_components,undefined);
// Importing an earlier subset leaves other cases intact and resets attestation.
const subset=structuredClone(saved);subset.annotations=subset.annotations.slice(0,1);ctx.subsetInput={files:[{text:async()=>JSON.stringify(subset)}]};
vm.runInContext('importDraft(sheets[0].sheet,subsetInput)',ctx);await new Promise(resolve=>setImmediate(resolve));
assert.match(ids.get('message').textContent,/已导入草稿/);assert.equal(vm.runInContext('sheets[0].sheet.annotations[1].components.specificity',ctx),3);
console.log('PASS: DOM render, explicit edit, export, bound import, and re-attestation; fixture only');})().catch(e=>{console.error(e);process.exitCode=1;});
''')
    result = subprocess.run([node, str(runner), str(fixture)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'PASS: DOM' in result.stdout
