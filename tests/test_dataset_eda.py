"""Проверки разметки, происхождения и утечек на синтетическом наборе."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

from src.dataset_eda import (prepare_paths,digest,inventory,update_class_mapping,find_duplicates,
                            apply_reviews,build_dataset,split_components,technical_metrics,perceptual_hash,
                            discover_images,reconcile_reviews,selected_present)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=prepare_paths(Path(self.temp.name))

    def synthetic(self):
        rows=[];mapping={f'moscow/district/poi_{i}':i for i in range(3)}
        for poi,k in mapping.items():
            for n in range(12):
                image_id=f'syn_{k}_{n:02}'
                path=Path('raw')/f'{image_id}.png';(self.root/path).parent.mkdir(exist_ok=True)
                Image.new('RGB',(32,32),(k*60,n*15,100)).save(self.root/path)
                rows.append(dict(image_id=image_id,file_path=path.as_posix(),poi_id=poi,initial_poi_id=poi,
                    readable=True,label_status='verified',sha256=digest(self.root/path),
                    width=32,height=32,duplicate_group=image_id,series_id='',quality_flags=''))
        return pd.DataFrame(rows),mapping

    def test_splits_deterministic_and_grouped(self):
        frame,mapping=self.synthetic()
        frame.loc[[0,12],'series_id']='shared_series'
        frame.loc[1,'duplicate_group']=frame.loc[2,'duplicate_group']
        result,ready=build_dataset(frame,self.root,mapping)
        self.assertTrue(ready)
        self.assertEqual(result.loc[[0,12],'split'].nunique(),1)
        self.assertEqual(result.loc[[1,2],'split'].nunique(),1)
        again,_=build_dataset(frame,self.root,mapping)
        self.assertTrue(result.split.equals(again.split))
        for split in ['train','val','test']:
            self.assertEqual(set(result[result.split.eq(split)].poi_id),set(mapping))
        for row in result.itertuples():self.assertEqual(digest(self.root/row.processed_path),row.sha256)

    def test_pending_never_released(self):
        frame,mapping=self.synthetic();frame['label_status']='needs_review'
        result,ready=build_dataset(frame,self.root,mapping)
        self.assertFalse(ready);self.assertTrue(result.split.eq('').all())
        self.assertTrue(pd.read_csv(self.root/'data/annotations/train.csv').empty)

    def test_enrichment_keeps_old_splits_and_quarantines_bridge(self):
        frame,mapping=self.synthetic()
        previous,ready=build_dataset(frame,self.root,mapping)
        self.assertTrue(ready)
        previous.to_csv(self.root/'data/annotations/split_anchors.csv',index=False)
        a=previous.index[previous.split.eq('train')][0]
        b=previous.index[previous.split.eq('test')][0]
        frame.loc[[a,b],'series_id']='new_bridge'
        result,ready=build_dataset(frame,self.root,mapping)
        self.assertTrue(ready)
        self.assertTrue(result.loc[[a,b],'split'].eq('').all())
        self.assertTrue(result.loc[[a,b],'exclusion_reason'].eq('cross_split_anchor_conflict').all())
        unaffected=~result.index.isin([a,b])
        self.assertTrue(result.loc[unaffected,'split'].equals(previous.loc[unaffected,'split']))

    def test_all_manual_policy_includes_high_count_poi(self):
        self.manual_fixture()
        (self.root/'config/dataset_sources.json').write_text('{"manual_selection":"all"}')
        descriptors,_,plan=discover_images(self.root)
        self.assertEqual(len(descriptors),38)
        self.assertEqual(plan.selected_manual.sum(),6)

    def test_external_overlap_is_not_released_or_relabelled(self):
        frame,mapping=self.synthetic()
        frame.iloc[[0]][['image_id']].to_csv(self.root/'data/annotations/external_overlap_quarantine.csv',index=False)
        result,ready=build_dataset(frame,self.root,mapping)
        self.assertTrue(ready)
        self.assertEqual(result.loc[0,'label_status'],'verified')
        self.assertEqual(result.loc[0,'split'],'')
        self.assertEqual(result.loc[0,'exclusion_reason'],'external_negative_overlap_candidate')

    def test_insufficient_groups_block_release(self):
        frame,mapping=self.synthetic();frame.loc[2:11,'label_status']='uncertain'
        result,ready=build_dataset(frame,self.root,mapping)
        self.assertFalse(ready);self.assertTrue(result.split.eq('').all())

    def test_exact_copy_keeps_one(self):
        frame,mapping=self.synthetic()
        frame=pd.concat([frame,frame.iloc[[0]].assign(image_id='copy')],ignore_index=True)
        result,ready=build_dataset(frame,self.root,mapping)
        self.assertTrue(ready)
        self.assertEqual(result.exclusion_reason.eq('exact_duplicate').sum(),1)
        self.assertTrue(result[result.split.ne('')].sha256.is_unique)

    def test_conflicting_verified_labels_excluded(self):
        frame,mapping=self.synthetic()
        copy=frame.iloc[[0]].assign(image_id='conflict',poi_id=list(mapping)[1])
        frame=pd.concat([frame,copy],ignore_index=True)
        result,_=build_dataset(frame,self.root,mapping)
        self.assertEqual(result.exclusion_reason.eq('exact_label_conflict').sum(),2)
        self.assertTrue(result[result.exclusion_reason.eq('exact_label_conflict')].split.eq('').all())

    def test_excluded_bridge_does_not_break_group(self):
        frame,_=self.synthetic()
        frame.loc[0,'duplicate_group']='bridge';frame.loc[1,'duplicate_group']='bridge'
        frame.loc[[1,2],'series_id']='series';frame.loc[1,'label_status']='rejected'
        groups=split_components(frame)
        self.assertEqual(groups[0],groups[2])

    def test_class_indices_append_without_renumber(self):
        old=update_class_mapping(self.root,['moscow/d/z','moscow/d/m'])
        new=update_class_mapping(self.root,['moscow/d/a','moscow/d/z'])
        for key,value in old.items():self.assertEqual(new[key],value)
        self.assertGreater(new['moscow/d/a'],max(old.values()))

    def test_technical_metrics(self):
        image=Image.new('RGB',(32,32),(120,120,120))
        result=technical_metrics(image)
        self.assertEqual(result['brightness'],120)
        self.assertEqual(result['sharpness'],0)
        self.assertEqual(len(perceptual_hash(image)),16)

    def test_metadata_portable_and_corruption_and_reviews(self):
        raw=self.root/'data/raw/wikimedia/moscow/district/poi';raw.mkdir(parents=True)
        Image.new('RGB',(40,30),'red').save(raw/'42.png')
        Image.new('RGB',(40,30),'blue').save(raw/'43.png')
        (raw/'44.jpg').write_bytes(b'corrupt image')
        (self.root/'config').mkdir()
        pd.DataFrame([dict(city='moscow',district='district',poi='poi',category='Example')]).to_csv(
            self.root/'config/wikimedia_pois.csv',index=False)
        metadata_path=self.root/'data/raw/wikimedia/metadata.csv'
        pd.DataFrame([dict(city='moscow',district='district',poi='poi',category='Example',
            file='Z:\\another-machine\\data\\raw\\wikimedia\\moscow\\district\\poi\\42.png',
            source_url='https://upload.wikimedia.org/wikipedia/commons/example.png',
            artist='<a href="https://commons.wikimedia.org/wiki/User:Example">Example Author</a>',
            license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',commons_title='File:Example.png')]).to_csv(metadata_path,index=False)
        before=digest(metadata_path)
        frame,hashes,mapping=inventory(self.root)
        self.assertEqual(digest(metadata_path),before)
        matched=frame[frame.file_path.str.endswith('42.png')].iloc[0]
        self.assertEqual(matched.source,'wikimedia_commons')
        self.assertEqual(matched.author,'Example Author');self.assertEqual(matched.licence,'CC BY 4.0')
        self.assertEqual(frame[frame.file_path.str.endswith('43.png')].author.iloc[0],'unknown')
        self.assertEqual(frame.source.eq('unknown').sum(),2)
        self.assertEqual((~frame.readable).sum(),1)
        frame,pairs=find_duplicates(frame,hashes,self.root)
        reviewed=apply_reviews(frame,self.root,mapping)
        self.assertEqual(reviewed.label_status.eq('verified').sum(),0)
        path=self.root/'data/annotations/review_edits.csv'
        edits=pd.read_csv(path,dtype=str,keep_default_na=False)
        idx=edits[edits.image_id.eq(matched.image_id)].index[0]
        edits.loc[idx,'label_status']='verified';edits.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'reviewer'):apply_reviews(frame,self.root,mapping)
        edits.loc[idx,'reviewer']='test reviewer';edits.to_csv(path,index=False)
        applied=apply_reviews(frame,self.root,mapping)
        self.assertEqual(applied.label_status.eq('verified').sum(),1)
        edits.loc[idx,'sha256']='changed';edits.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'sha256'):apply_reviews(frame,self.root,mapping)

    def test_exact_and_commons_identity_cross_poi(self):
        frame=pd.DataFrame([
            dict(image_id='a',sha256='same',commons_file_id='1',readable=False,poi_id='poi_a'),
            dict(image_id='b',sha256='same',commons_file_id='1',readable=False,poi_id='poi_b'),
            dict(image_id='c',sha256='different',commons_file_id='1',readable=False,poi_id='poi_b')])
        result,pairs=find_duplicates(frame,{},self.root)
        self.assertEqual(result.duplicate_group.nunique(),1)
        self.assertTrue(pairs.cross_poi.any())
        self.assertEqual(set(pairs.kind),{'sha256_exact','commons_same_file'})

    def manual_fixture(self):
        (self.root/'config').mkdir()
        pd.DataFrame([dict(city='moscow',district='d',poi=slug,category=slug) for slug in ['low','high']]).to_csv(
            self.root/'config/wikimedia_pois.csv',index=False)
        for slug,count in [('low',2),('high',30)]:
            folder=self.root/'data/raw/wikimedia/moscow/d'/slug;folder.mkdir(parents=True)
            for n in range(count):Image.new('RGB',(16,16),(n,40,80)).save(folder/f'{n}.png')
        pd.DataFrame(columns=['city','district','poi','file','source_url','artist','license','license_url','commons_title']).to_csv(
            self.root/'data/raw/wikimedia/metadata.csv',index=False)
        records=[]
        for slug in ['low','high','new']:
            folder=self.root/'data/raw/manual'/slug;folder.mkdir(parents=True)
            for n in range(2):Image.new('RGB',(16,16),(50,n,120)).save(folder/f'{n}.png')
            records.append(dict(folder=slug,city='moscow',district='d',poi=slug,author='Shilkova A.M.',licence='Public Domain'))
        pd.DataFrame(records).to_csv(self.root/'config/manual_pois.csv',index=False)

    def test_manual_selection_rule_and_attribution(self):
        self.manual_fixture()
        descriptors,_,plan=discover_images(self.root)
        self.assertEqual(len(descriptors),36)
        self.assertEqual(plan.set_index('poi_id').loc['moscow/d/high','selected_manual'],0)
        self.assertEqual(plan.set_index('poi_id').loc['moscow/d/low','selected_manual'],2)
        self.assertEqual(plan.set_index('poi_id').loc['moscow/d/new','selected_manual'],2)
        frame,_,_=inventory(self.root)
        manual=frame[frame.source.eq('manual')]
        self.assertEqual(len(manual),4)
        self.assertTrue(manual.author.eq('Shilkova A.M.').all())
        self.assertTrue(manual.licence.eq('Public Domain').all())

    def test_reconcile_keeps_deleted_history_and_provenance(self):
        self.manual_fixture()
        frame,_,mapping=inventory(self.root)
        reconcile_reviews(frame,self.root,confirm_manual=True)
        removed=frame[frame.file_path.str.endswith('wikimedia/moscow/d/low/0.png')].iloc[0]
        (self.root/removed.file_path).unlink()
        current,_,mapping=inventory(self.root)
        edits=reconcile_reviews(current,self.root,confirm_wikimedia=True,confirm_manual=True)
        self.assertEqual(len(edits),36)
        deleted=edits[edits.image_id.eq(removed.image_id)].iloc[0]
        self.assertEqual(deleted.label_status,'rejected')
        self.assertEqual(deleted.rejection_reason,'removed_by_user')
        self.assertFalse(current[current.image_id.eq(removed.image_id)].file_exists.iloc[0])
        reviewed=apply_reviews(current,self.root,mapping)
        self.assertEqual(reviewed[reviewed.image_id.eq(removed.image_id)].rejection_reason.iloc[0],'removed_by_user')
        annotations=pd.read_csv(self.root/'data/annotations/annotations.csv',keep_default_na=False)
        self.assertTrue({'source','author','licence'}<=set(annotations))
        self.assertTrue(annotations[annotations.source.eq('manual')].label_status.eq('verified').all())

    def test_missing_source_never_enters_splits(self):
        frame,mapping=self.synthetic()
        frame['file_exists']=True;frame['selected_for_dataset']=True;frame['rejection_reason']=''
        frame.loc[0,['file_exists','selected_for_dataset','label_status','rejection_reason']]=[
            False,False,'rejected','removed_by_user']
        (self.root/frame.loc[0,'file_path']).unlink()
        result,ready=build_dataset(frame,self.root,mapping)
        self.assertTrue(ready)
        self.assertEqual(result.loc[0,'split'],'')
        self.assertEqual(result.loc[0,'exclusion_reason'],'removed_by_user')


if __name__=='__main__':unittest.main()
