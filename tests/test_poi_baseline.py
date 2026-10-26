"""Проверки целевой верификации и калибровки без скачивания/обработки датасета."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.poi_baseline import (l2_normalize, select_gallery, poi_scores, acceptance_matrix,
                              calibrate_policy, evaluate, load_splits, safe_project_path,
                              create_encoder, POIVerifier, PREPROCESSING_VERSION)


class POIBaselineTests(unittest.TestCase):
    def config(self):
        return {'target_known_wrong_far':0.01, 'target_hard_wrong_far':0.05,
                'score_grid_size':11, 'margin_grid_size':11}

    def test_scores_compare_target_to_its_own_references(self):
        scores, neighbours = poi_scores(np.array([[1., 0.]]), np.array([[1., 0.], [0., 1.]]), ['a', 'b'], ['b', 'a'])
        np.testing.assert_allclose(scores, [[0., 1.]])
        np.testing.assert_array_equal(neighbours, [[1, 0]])
        accepted = acceptance_matrix(scores, {'score_threshold':.8, 'margin_threshold':.1})
        np.testing.assert_array_equal(accepted, [[False, True]])

    def test_relative_margin_and_tie_reject(self):
        scores = np.array([[.9, .89], [.9, .9], [.9, .3]])
        accepted = acceptance_matrix(scores, {'score_threshold':.8, 'margin_threshold':.05})
        np.testing.assert_array_equal(accepted, [[False, False], [False, False], [True, False]])

    def test_low_score_reject_even_if_target_is_best(self):
        self.assertFalse(acceptance_matrix(np.array([[.4, .1]]), {'score_threshold':.8, 'margin_threshold':0.}).any())

    def test_calibration_has_reject_all_candidate_for_float32(self):
        # Одинаковые уверенные scores для верной и неверной цели: FAR=0
        # достигается лишь отказом; nextafter не должен округлиться в float32.
        scores = np.array([[.9, .2], [.9, .2]], dtype=np.float32)
        config = self.config(); config['target_hard_wrong_far'] = 0.
        policy, _ = calibrate_policy(scores, np.array([0, 1]), config)
        self.assertFalse(acceptance_matrix(scores, policy).any())

    def test_calibration_constraints_and_determinism(self):
        scores = np.array([[.95,.4],[.3,.96],[.65,.7],[.8,.2]])
        truth = np.array([0,1,0,0])
        a, grid = calibrate_policy(scores, truth, self.config())
        b, _ = calibrate_policy(scores, truth, self.config())
        self.assertEqual(a,b)
        self.assertEqual(a['calibration_split'],'val')
        self.assertLessEqual(a['known_wrong_far'],.01)
        self.assertLessEqual(a['hard_wrong_far'],.05)
        self.assertGreaterEqual(a['true_accept_rate'],.75)
        self.assertFalse(a['external_unknown_calibrated'])
        self.assertTrue(len(grid)>1)

    def test_gallery_diverse_deterministic_and_one_group_per_poi(self):
        train = pd.DataFrame({'image_id':['a1','a2','a3','b1','b2'], 'poi_id':['a','a','a','b','b'],
                              'split_group':['same','same','other','b1','b2']})
        vectors = l2_normalize(np.array([[1,.1],[1,.11],[1,-.9],[-1,.1],[-1,-.8]]))
        gallery, values = select_gallery(train,vectors,20)
        second, values2 = select_gallery(train,vectors,20)
        self.assertEqual(len(gallery),4)
        self.assertTrue(set(gallery.image_id)<=set(train.image_id))
        self.assertFalse(gallery.duplicated(['poi_id','split_group']).any())
        pd.testing.assert_frame_equal(gallery,second)
        np.testing.assert_array_equal(values,values2)

    def test_metrics_distinguish_top1_and_verification_and_negative_scope(self):
        frame = pd.DataFrame({'image_id':['one','two'], 'poi_id':['a','b'], 'source':['manual','wikimedia_commons']})
        metrics, per_poi, pairs, confusion = evaluate(np.array([[.95,.2],[.2,.7]]),frame,['a','b'],
                                                    {'score_threshold':.8,'margin_threshold':.1})
        self.assertEqual(metrics['top1_accuracy'],1.)
        self.assertEqual(metrics['true_accept_rate'],.5)
        self.assertEqual(metrics['known_wrong_far'],0.)
        self.assertEqual(metrics['hard_wrong_far'],0.)
        self.assertIsNone(metrics['open_set_far'])
        self.assertEqual(len(pairs),4)
        self.assertEqual(int(confusion.to_numpy().sum()),2)

    def test_invalid_embeddings_and_missing_reference_fail(self):
        for array in (np.array([[0.,0.]]), np.array([[np.nan,1.]])):
            with self.assertRaises(ValueError): l2_normalize(array)
        with self.assertRaises(ValueError):
            poi_scores(np.array([[1.,0.]]),np.array([[1.,0.]]),['a'],['a','b'])

    def test_split_integrity_rejects_sha_and_series_leaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root/'data/annotations'; annotations.mkdir(parents=True)
            (annotations/'class_mapping.json').write_text(json.dumps({'a':0,'b':1}))
            for name in ['train','val','test']:
                rows = []
                for poi,index in [('a',0),('b',1)]:
                    uid = name+poi
                    rows.append(dict(image_id=uid,file_path=uid+'.jpg',poi_id=poi,class_index=index,
                                     label_status='verified',sha256=uid,duplicate_group=uid,split_group=uid,
                                     source='manual',author='Shilkova A.M.',licence='Public Domain',series_id=''))
                pd.DataFrame(rows).to_csv(annotations/f'{name}.csv',index=False)
            splits,_ = load_splits(root)
            frame = splits['val'].copy(); frame.loc[0,'sha256']='traina'
            frame.to_csv(annotations/'val.csv',index=False)
            with self.assertRaisesRegex(ValueError,'sha256'): load_splits(root)
            frame.loc[0,'sha256']='vala'; frame.loc[0,'series_id']='shared'
            frame.to_csv(annotations/'val.csv',index=False)
            train = splits['train'].copy(); train.loc[0,'series_id']='shared'
            train.to_csv(annotations/'train.csv',index=False)
            with self.assertRaisesRegex(ValueError,'series_id'): load_splits(root)

    def test_paths_confined_to_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): safe_project_path(Path(tmp),'../outside.jpg')

    def test_weights_never_fallback_to_random(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.pth';path.write_bytes(b'not pretrained')
            with self.assertRaisesRegex(ValueError,'SHA-256'): create_encoder(path)

    def test_unsupported_target_refuses_without_reading_image(self):
        verifier = POIVerifier.__new__(POIVerifier)
        verifier.poi_ids=['a','b']
        verifier.metadata={'reference_version':'ref','threshold_version':'threshold'}
        result=verifier.verify(Path('does-not-exist.jpg'),'unknown')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'],'target_poi_not_supported')
        self.assertIsNone(result['score'])
        self.assertEqual(result['preprocessing_version'],PREPROCESSING_VERSION)

    def test_tampered_artifact_refused_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'imagenet_weights.pth').write_bytes(b'changed')
            (root/'metadata.json').write_text(json.dumps({'schema_version':1,
                'preprocessing_version':PREPROCESSING_VERSION,'artifact_sha256':{'imagenet_weights.pth':'bad'}}))
            with self.assertRaisesRegex(ValueError,'Артефакт изменён'): POIVerifier(root)


if __name__=='__main__':
    unittest.main()
