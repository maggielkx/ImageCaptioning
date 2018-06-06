import vist
from datetime import datetime
import os.path as osp
from imageio import imwrite


import warnings
warnings.filterwarnings("error", category=UserWarning)

vist_images_dir = 'images/'
vist_annotations_dir = 'annotations/'
sis = vist.Story_in_Sequence(vist_images_dir, vist_annotations_dir)
dii = vist.Description_in_Isolation(vist_images_dir, vist_annotations_dir)

# album_id = sis.Albums.keys()[0]
# # sis.show_album(album_id)
# album = sis.Albums[album_id]
# # pprint(album)
#
#
# story_ids = sis.Albums[album_id]['story_ids']
# story_id = story_ids[0]
# # sis.show_story(story_id)
# print sis.Stories[story_id]['img_ids']


# check story order
def check_dts_order(dts):
    flag = True
    for i in range(1, len(dts)):
        if dts[i] <= dts[i-1]:
            flag = False
    return flag

inorder = 0
stories_ordered = []
for story in sis.stories:
    dts = []
    for i, sent_id in enumerate(story['sent_ids']):
        sent = sis.Sents[sent_id]
        assert sent['order'] == i
        img = sis.Images[sent['img_id']]
        dt = datetime.strptime(img['datetaken'], '%Y-%m-%d %H:%M:%S')
        dts += [dt]
    if check_dts_order(dts):
        inorder += 1
        stories_ordered.append(story)

print 'Among %s stories, %s [%.2f%%] are in order' % (len(sis.stories), inorder, inorder*100.0/len(sis.stories))

import json
f = open('vist.json', 'w')
f_vist ={}

f_train = open('vist_train.json', 'w')
f_vist_train ={}

f_val = open('vist_val.json', 'w')
f_vist_val ={}

f_test = open('vist_test.json', 'w')
f_vist_test ={}

for n, story in enumerate(stories_ordered):
    story_info ={}

    story_id = story['id']
    print 'Reading {}th story: {}'.format(n, story_id)
    if int(story_id) in [9515, 9516, 9517, 9518, 9519]:
        continue
    story = sis.Stories[story_id]
    sent_ids = story['sent_ids']
    img_ids = story['img_ids']
    album_id = story['album_id']
    split = sis.Albums[album_id]['split']

    sents = []
    captions_pack = []
    for i, (sent_id, img_id) in enumerate(zip(sent_ids, img_ids)):

        img = sis.Images[img_id]
        img_name = osp.join(sis.images_dir, 'resizedStory', split, img_id + '.jpg')
        img_file = osp.join(sis.images_dir, split, img_id + '.jpg')

        if osp.exists(img_file) and len(dii.Images[img_id]['sent_ids'])>0:
            try:
                img_content = sis.read_img(img_file)
                imwrite(img_name, img_content)

                sent = sis.Sents[sent_id]
                sentence = sent['text']
                sents.append(sentence)

                captions = []
                caption_ids = dii.Images[img_id]['sent_ids']
                for k, caption_id in enumerate(caption_ids):
                    caption = dii.Sents[caption_id]['text']
                    captions.append(caption)
                captions_pack.append(captions)

            except UserWarning or IOError or ValueError:
                continue
    if len(captions_pack) == 5:
        story_info['album_id']=album_id
        story_info['img_ids']=img_ids
        story_info['sents']=sents
        story_info['captions']=captions_pack
        story_info['split'] = split
        if split == 'train':
            f_vist[story_id] = story_info
            f_vist_train[story_id] = story_info
        elif split == 'val':
            f_vist[story_id] = story_info
            f_vist_val[story_id] = story_info
        else:
            f_vist[story_id] = story_info
            f_vist_test[story_id] = story_info
        print('-' * 30)
    # else:
    #     print('Skip story {}'.format(story_id))

json.dump(f_vist, f)
f.close()

json.dump(f_vist_train, f_train)
f_train.close()
json.dump(f_vist_val, f_val)
f_val.close()
json.dump(f_vist_test, f_test)
f_test.close()


import json


def sort_dataset(ids):
    idx_train=[]
    idx_val=[]
    idx_test=[]

    for i, id in enumerate(ids):
        if vist[id]['split'] == 'train':
            idx_train.append(id)
        elif vist[id]['split'] == 'val':
            idx_val.append(id)
        else:
            idx_test.append(id)
    print(len(idx_train), len(idx_val), len(idx_test))

    return idx_train, idx_val, idx_test

vist = json.load(open('vist.json', 'r'))
ids = vist.keys()
idx_train, idx_val, idx_test = sort_dataset(ids)


# import numpy as np
# idx_train2val = np.random.choice(idx_train, 2, replace=False)
# idx_train = list(set(idx_train)-set(idx_train2val))
# idx_train2test = np.random.choice(idx_train, 1, replace=False)
# idx_train = list(set(idx_train)-set(idx_train2test))
#
#
# import shutil
# def copy_from_train(idxlist, split):
#     for id in idxlist:
#         for img_id in vist[id]['img_ids']:
#             img_name = osp.join(vist_images_dir, 'resizedStory', 'train', img_id + '.jpg')
#             img_dest = osp.join(vist_images_dir, 'resizedStory', split, img_id + '.jpg')
#             shutil.copy(img_name, img_dest)
#         vist[id]['split'] = unicode(split)
#
#
# copy_from_train(idx_train2val, 'val')
# copy_from_train(idx_train2test, 'test')
#
# sort_dataset(ids)