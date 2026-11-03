"""Проверки знака эффекта и недопустимости подмены сеансов кадрами."""
import unittest
import numpy as np
from src.viewpoint_statistics import contrasts,clustered_bootstrap,macro_f1


class ViewpointStatisticsTests(unittest.TestCase):
    def test_large_drop_is_not_small_improvement(self):
        y=np.array([0,0,1,1]);correct=y.copy();wrong=1-y
        p={key:np.array([value]*3) for key,value in dict(
            small_O=wrong,large_O=correct,small_R=wrong,large_R=wrong).items()}
        out=contrasts(y,p,2)
        self.assertEqual(out['C'],1.)
        self.assertEqual(out['small_gain'],0.)
        self.assertEqual(out['large_gain'],-1.)

    def test_unknown_sessions_do_not_produce_interval(self):
        y=np.array([0,1]);p={key:np.array([y]*3) for key in ('small_O','large_O','small_R','large_R')}
        rows=[dict(image_id=str(i),poi_id=str(i),session_id='',outing_id='') for i in range(2)]
        result=clustered_bootstrap(y,p,rows,2,5000,2026)
        self.assertEqual(result['performed_replicates'],0)
        self.assertIsNone(result['C_ci95'])
        self.assertFalse(result['hypothesis_confirmed'])

    def test_joint_outings_and_fixed_mapping(self):
        y=np.array([0,0,1,1]);p={key:np.array([y]*3) for key in ('small_O','large_O','small_R','large_R')}
        rows=[dict(image_id=str(i),poi_id=str(i//2),session_id='',outing_id=str(i%2)) for i in range(4)]
        a=clustered_bootstrap(y,p,rows,2,5000,2026)
        self.assertEqual(a['performed_replicates'],5000)
        self.assertEqual(a['C_ci95'],[0.,0.])
        self.assertEqual(a['point']['F']['small_O'],1.)
        self.assertFalse(a['hypothesis_confirmed'])
        # Отсутствующий в срезе класс не исчезает из полного mapping.
        self.assertEqual(macro_f1([0],[0],2),.5)


if __name__=='__main__':unittest.main()
