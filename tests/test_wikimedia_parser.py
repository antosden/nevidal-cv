"""Проверка CSV и каталогов без запросов к Wikimedia."""
import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock, patch
import requests

from scripts import wikimedia_parser as parser


class WikimediaParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, text):
        path = self.root / "pois.csv"
        path.write_text(text, encoding="utf-8-sig")
        return path

    def test_csv_bom_and_quoted_category(self):
        configs = parser.load_pois(self.config(
            'city,district,poi,category\nmoscow,ostankinskiy,new_poi,"Category:Example, facade"\n'
        ))
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].category, "Example, facade")
        self.assertEqual(configs[0].folder(self.root), self.root / "moscow/ostankinskiy/new_poi")

    def test_same_poi_in_different_districts(self):
        configs = parser.load_pois(self.config(
            'city,district,poi,category\nmoscow,one,tower,One\nmoscow,two,tower,Two\n'
        ))
        self.assertEqual(len(configs), 2)
        self.assertNotEqual(configs[0].folder(self.root), configs[1].folder(self.root))

    def test_invalid_configuration(self):
        cases = [
            'poi,category\ntower,Tower\n',
            'city,district,poi,category\n',
            'city,district,poi,category\nmoscow,../escape,tower,Tower\n',
            'city,district,poi,category\nmoscow,/absolute,tower,Tower\n',
            'city,district,poi,category\nmoscow,one,con,Tower\n',
            'city,district,poi,category\nmoscow,one,tower,\n',
            'city,district,poi,category\nmoscow,one,tower,One\nmoscow,one,tower,Two\n',
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parser.load_pois(self.config(text))

    def test_default_csv(self):
        configs = parser.load_pois(parser.DEFAULT_CONFIG)
        self.assertTrue(configs)
        identities = {(c.city, c.district, c.poi) for c in configs}
        self.assertEqual(len(identities), len(configs))
        self.assertTrue(all(c.city and c.district and c.poi and c.category for c in configs))

    def test_download_uses_csv_hierarchy_and_metadata(self):
        configs = parser.load_pois(self.config(
            'city,district,poi,category\nmoscow,ostankinskiy,new_poi,New Category\n'
        ))
        page = {'pageid': 42, 'title': 'File:Example.jpg'}
        info = {'mime': 'image/jpeg', 'url': 'https://example.invalid/42.jpg',
                'width': 500, 'height': 300, 'extmetadata': {
                    'Artist': {'value': 'Example author'},
                    'LicenseShortName': {'value': 'CC BY-SA 4.0'},
                    'LicenseUrl': {'value': 'https://example.invalid/license'},
                }}
        session = MagicMock()
        response = session.get.return_value.__enter__.return_value
        response.iter_content.return_value = [b'fake image bytes']
        output = self.root / 'output'
        with patch.object(parser, 'get_files', return_value=iter([(page, info)])) as get_files, \
                patch.object(parser.time, 'sleep'):
            rows = parser.download_pois(configs, output, session, max_per_poi=1)
        get_files.assert_called_once_with('New Category', session)
        expected = output / 'moscow/ostankinskiy/new_poi/42.jpg'
        self.assertEqual(expected.read_bytes(), b'fake image bytes')
        self.assertEqual(rows[0]['file'], str(expected))
        with (output / 'metadata.csv').open(encoding='utf-8', newline='') as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual(saved['city'], 'moscow')
        self.assertEqual(saved['district'], 'ostankinskiy')
        self.assertEqual(saved['poi'], 'new_poi')
        self.assertEqual(saved['artist'], 'Example author')
        self.assertEqual(saved['license'], 'CC BY-SA 4.0')

    def test_preserve_existing_metadata(self):
        path = self.root / 'metadata.csv'
        path.write_text('file,poi,artist,license,custom\nold.jpg,old,Author,CC BY,keep\n', encoding='utf-8')
        parser.save_metadata(self.root, [{'file': 'new.jpg', 'poi': 'new', 'city': 'moscow'}])
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['artist'], 'Author')
        self.assertEqual(rows[0]['custom'], 'keep')

    def test_nonpositive_limit_has_no_side_effects(self):
        output = self.root / 'not-created'
        with self.assertRaises(ValueError):
            parser.download_pois([], output, Mock(), max_per_poi=0)
        self.assertFalse(output.exists())

    def entry(self, page_id):
        return ({'pageid':page_id,'title':f'File:{page_id}.jpg'},
                {'mime':'image/jpeg','url':f'https://example.invalid/{page_id}.jpg'})

    def session(self):
        session=MagicMock()
        session.get.return_value.__enter__.return_value.iter_content.return_value=[b'complete image']
        return session

    def progress(self, path):
        with path.open(encoding='utf-8-sig',newline='') as stream:
            return list(csv.DictReader(stream))

    def test_resume_after_interrupted_category(self):
        path=self.config('city,district,poi,category\nmoscow,one,tower,Tower\n')
        output=self.root/'output'
        def interrupted():
            yield self.entry(1)
            raise RuntimeError('API connection lost')
        first=self.session()
        with patch.object(parser,'get_files',return_value=interrupted()),patch.object(parser.time,'sleep'):
            with self.assertRaises(RuntimeError):
                parser.download_pois(parser.load_pois(path),output,first,2,config_path=path)
        self.assertEqual(self.progress(path)[0]['status'],'error')
        self.assertEqual(self.progress(path)[0]['downloaded_count'],'1')
        self.assertIn('API connection lost',self.progress(path)[0]['last_error'])
        with (output/'metadata.csv').open(encoding='utf-8',newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),1)
        second=self.session()
        with patch.object(parser,'get_files',return_value=iter([self.entry(1),self.entry(2)])),patch.object(parser.time,'sleep'):
            parser.download_pois(parser.load_pois(path),output,second,2,config_path=path)
        second.get.assert_called_once_with('https://example.invalid/2.jpg',stream=True,timeout=60)
        self.assertEqual(self.progress(path)[0]['status'],'done')
        self.assertEqual(self.progress(path)[0]['downloaded_count'],'2')
        self.assertEqual(self.progress(path)[0]['last_error'],'')

    def test_done_skips_even_category_query(self):
        path=self.config('city,district,poi,category\nmoscow,one,tower,Tower\n')
        output=self.root/'output'
        with patch.object(parser,'get_files',return_value=iter([self.entry(1)])),patch.object(parser.time,'sleep'):
            parser.download_pois(parser.load_pois(path),output,self.session(),1,config_path=path)
        session=self.session()
        with patch.object(parser,'get_files') as api:
            parser.download_pois(parser.load_pois(path),output,session,1,config_path=path)
        api.assert_not_called();session.get.assert_not_called()

    def test_recovers_file_saved_before_metadata_checkpoint(self):
        path=self.config('city,district,poi,category\nmoscow,one,tower,Tower\n')
        configs=parser.load_pois(path);output=self.root/'output'
        folder=configs[0].folder(output);folder.mkdir(parents=True)
        (folder/'1.jpg').write_bytes(b'previous complete download')
        session=self.session()
        with patch.object(parser,'get_files',return_value=iter([self.entry(1)])),patch.object(parser.time,'sleep'):
            parser.download_pois(configs,output,session,1,config_path=path)
        session.get.assert_not_called()
        self.assertEqual(self.progress(path)[0]['status'],'done')
        self.assertTrue((output/'metadata.csv').is_file())

    def test_failed_download_is_not_done(self):
        path=self.config('city,district,poi,category\nmoscow,one,tower,Tower\n')
        session=self.session();session.get.side_effect=requests.ConnectionError('offline')
        with patch.object(parser,'get_files',return_value=iter([self.entry(1)])):
            parser.download_pois(parser.load_pois(path),self.root/'output',session,1,config_path=path)
        self.assertEqual(self.progress(path)[0]['status'],'error')
        self.assertEqual(self.progress(path)[0]['downloaded_count'],'0')

    def test_increased_limit_resumes_done_poi(self):
        path=self.config('city,district,poi,category\nmoscow,one,tower,Tower\n')
        output=self.root/'output'
        with patch.object(parser,'get_files',return_value=iter([self.entry(1)])),patch.object(parser.time,'sleep'):
            parser.download_pois(parser.load_pois(path),output,self.session(),1,config_path=path)
        session=self.session()
        with patch.object(parser,'get_files',return_value=iter([self.entry(1),self.entry(2)])),patch.object(parser.time,'sleep'):
            parser.download_pois(parser.load_pois(path),output,session,2,config_path=path)
        session.get.assert_called_once_with('https://example.invalid/2.jpg',stream=True,timeout=60)
        self.assertEqual(self.progress(path)[0]['downloaded_count'],'2')

    def test_progress_preserves_other_rows_and_custom_columns(self):
        path=self.config('city,district,poi,category,notes\nmoscow,one,tower,Tower,keep\nmoscow,two,arch,Arch,other\n')
        config=parser.load_pois(path)[0]
        parser.save_progress(path,config,'done',1,'key')
        rows=self.progress(path)
        self.assertEqual(rows[0]['notes'],'keep')
        self.assertEqual(rows[1]['notes'],'other')
        self.assertEqual(rows[1]['status'],'')

    def test_invalid_progress(self):
        for suffix in ['unknown,0','done,-1','pending,abc']:
            with self.subTest(suffix=suffix),self.assertRaises(ValueError):
                parser.load_pois(self.config('city,district,poi,category,status,downloaded_count\nmoscow,one,tower,Tower,'+suffix+'\n'))


if __name__ == '__main__':
    unittest.main()
