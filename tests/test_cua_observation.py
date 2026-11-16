import copy
import unittest

from cua_defense.observation import model_image, stable_task


class ObservationTests(unittest.TestCase):
    def test_mask_keeps_coordinates_body_and_raw(self):
        from PIL import Image
        raw = Image.new('RGB', (1920,1088), (13,71,209))
        result = model_image(raw)
        self.assertEqual(result.size, raw.size)
        self.assertEqual(result.getpixel((1800,31)), (0,0,0))
        self.assertEqual(result.getpixel((1800,32)), raw.getpixel((1800,32)))
        self.assertEqual(raw.getpixel((1800,31)), (13,71,209))
        self.assertEqual(result.crop((0,32,1920,1088)).tobytes(), raw.crop((0,32,1920,1088)).tobytes())

    def test_dimensions_fail_closed(self):
        from PIL import Image
        with self.assertRaises(ValueError):
            model_image(Image.new('RGB',(1280,720)))

    def test_task_preparation_does_not_mutate_original(self):
        row={'metadata':{'config':[],'others':{}},'instruction':'legitimate goal'}
        before=copy.deepcopy(row)
        new=stable_task(row)
        self.assertEqual(row,before)
        self.assertEqual(new['instruction'],row['instruction'])
        self.assertEqual(len(new['metadata']['config']),1)


if __name__=='__main__':
    unittest.main()
