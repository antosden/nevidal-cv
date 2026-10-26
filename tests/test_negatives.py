"""Защита от утечек, смешения целей и неправильной калибровки negatives."""
import unittest
import numpy as np
import pandas as pd

from src.negative_dataset import split_negative_pois, duplicate_links
from src.negative_baseline import evaluate_external
from src.poi_baseline import calibrate_policy, acceptance_matrix


class NegativeTests(unittest.TestCase):
    def test_poi_and_duplicate_components_never_cross_splits(self):
        rows=[]
        for poi in range(6):
            for n in range(4):
                uid=f'{poi}_{n}'
                rows.append(dict(image_id=uid,sha256=uid,commons_file_id=uid,
                    source_poi_id=f'poi_{poi}',negative_type='building' if poi<4 else 'tower',
                    duplicate_group='shared' if (poi,n) in [(0,0),(1,0)] else uid,
                    active=n!=3,split=''))
        frame=pd.DataFrame(rows)
        result=split_negative_pois(frame)
        second=split_negative_pois(frame)
        pd.testing.assert_frame_equal(result,second)
        active=result[result.active]
        self.assertEqual(set(active.split),{'val','test'})
        self.assertEqual(active.groupby('source_poi_id').split.nunique().max(),1)
        self.assertEqual(active.loc[active.source_poi_id.isin(['poi_0','poi_1']),'split'].nunique(),1)
        self.assertTrue(result.loc[~result.active,'split'].eq('').all())

    def test_no_valid_negative_split_if_one_connected_poi(self):
        frame=pd.DataFrame([dict(image_id=str(i),source_poi_id='one',duplicate_group=str(i),
                                 negative_type='tower',active=True,split='') for i in range(3)])
        with self.assertRaisesRegex(ValueError,'Недостаточно'):split_negative_pois(frame)

    def test_hash_and_commons_cross_source_copies_are_linked(self):
        frame=pd.DataFrame({'image_id':['pos','neg1','neg2','other'],
            'sha256':['same','same','different','other'],
            'commons_file_id':['42','42','42','43']})
        uf,pairs=duplicate_links(frame,{})
        self.assertEqual(uf.find('pos'),uf.find('neg1'))
        self.assertEqual(uf.find('pos'),uf.find('neg2'))
        self.assertNotEqual(uf.find('pos'),uf.find('other'))
        self.assertGreaterEqual(len(pairs),2)

    def test_perceptual_copies_link_even_if_different_bytes(self):
        frame=pd.DataFrame({'image_id':['a','b','c'],'sha256':['a','b','c'],
                            'commons_file_id':['1','2','3']})
        rng=np.random.default_rng(42)
        hashes={uid:[f'{int(value):016x}' for value in rng.integers(0,2**63,size=11,dtype=np.uint64)]
                for uid in frame.image_id}
        hashes['b'][0]=hashes['a'][0]
        uf,pairs=duplicate_links(frame,hashes)
        self.assertEqual(uf.find('a'),uf.find('b'))
        self.assertNotEqual(uf.find('a'),uf.find('c'))

    def test_external_val_constrains_policy_without_changing_backbone(self):
        config={'target_known_wrong_far':.01,'target_hard_wrong_far':.05,
                'target_external_far':0.,'score_grid_size':21,'margin_grid_size':21}
        positives=np.array([[.95,.1],[.1,.95],[.65,.2],[.2,.65]])
        truth=np.array([0,1,0,1])
        negative_val=np.array([[.8,.2],[.2,.8]])
        old,_=calibrate_policy(positives,truth,config)
        policy,grid=calibrate_policy(positives,truth,config,negative_val)
        self.assertTrue(acceptance_matrix(negative_val,old).any())
        self.assertFalse(acceptance_matrix(negative_val,policy).any())
        self.assertTrue(policy['external_unknown_calibrated'])
        self.assertEqual(policy['external_calibration_split'],'negative_val')
        self.assertEqual(policy['external_any_poi_far'],0.)
        self.assertGreaterEqual(policy['true_accept_rate'],.5)
        self.assertIn('external_any_poi_far',grid)

    def test_external_far_denominator_is_images_not_number_of_targets(self):
        scores=np.array([[.95,.2],[.1,.3]])
        frame=pd.DataFrame({'image_id':['x','y'],'negative_type':['tower','tower'],
                            'source_poi_id':['one','two']})
        summary,queries,groups=evaluate_external(scores,frame,['a','b'],
                            {'score_threshold':.8,'margin_threshold':.05})
        self.assertEqual(summary['any_poi_far'],.5)
        self.assertEqual(summary['uniform_target_pair_far'],.25)
        self.assertEqual(summary['rejection_rate'],.5)
        self.assertEqual(int(queries.accepted_any_poi.sum()),1)

    def test_empty_external_validation_is_not_silently_ignored(self):
        cfg={'target_known_wrong_far':.01,'target_hard_wrong_far':.05,
             'score_grid_size':11,'margin_grid_size':11}
        with self.assertRaises(ValueError):calibrate_policy(np.array([[.9,.1]]),np.array([0]),cfg,np.empty((0,2)))


if __name__=='__main__': unittest.main()
