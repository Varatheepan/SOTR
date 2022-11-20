#################################################
    # Kanagarajah Sathursan
    # csv file writer for CORe50 dataset
    # writing order

    # for train data
    # image_path, session_label, object_label, category_label, session_object_task_id, session_category_task_id, image_label

    # for test data
    # image_path, session_label, object_label, category_label, image_label
#################################################
import os
import csv

images_dir = '/content/core50_128x128/'
train_ses_list = ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']
test_ses_list = ['s3', 's7', 's10']
objects_list = ['o' + str(i) for i in range(1,51)]
train_session_category_task_list = [[ses, cate] for ses in [1, 2, 4, 5, 6, 8, 9, 11] for cate in range(1, 11)]
train_session_object_task_list = [[ses, obj] for ses in [1, 2, 4, 5, 6, 8, 9, 11] for obj in range(1, 51)]

#################################################
    # writing train set as csv file
#################################################
with open(images_dir + 'train_image_path_and_labels-1.csv', 'w') as train_csvfile:

    filewriter = csv.writer(train_csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['image_path', 'session_label', 'object_label', 'category_label', 'session_object_task_id', 'session_category_task_id', 'image_label', 'index'])
    for i, ses in enumerate(train_ses_list):

        for j, obj in enumerate(objects_list):
            img_name_list = os.listdir(os.path.join(images_dir, ses, obj))
            img_name_list.sort()

            for k, img_name in enumerate(img_name_list):
                index = str(i * 300 * 50 + j * 300 + k)
                session_label = int(img_name[2:4])
                object_label = int(img_name[5:7])
                img_label = int(img_name[8:11])
                category_label = ((object_label - 1) // 5) + 1
                ses_obj_task_id = train_session_object_task_list.index([session_label, object_label]) + 1
                ses_cate_task_id = train_session_category_task_list.index([session_label, category_label]) + 1
                
                img_dir = '/content/core50_128x128/' + ses + '/' + obj + '/' + img_name

                filewriter.writerow([img_dir, session_label, object_label, category_label, ses_obj_task_id, ses_cate_task_id, img_label, index])

#################################################
    # writing test set as csv file
#################################################
with open(images_dir + 'test_image_path_and_labels-1.csv', 'w') as test_csvfile:

    filewriter = csv.writer(test_csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['image_path', 'session_label', 'object_label', 'category_label', 'image_label'])
    for ses in test_ses_list:

        for obj in objects_list:
            img_name_list = os.listdir(os.path.join(images_dir, ses, obj))
            img_name_list.sort()

            for img_name in img_name_list:

                session_label = int(img_name[2:4])
                object_label = int(img_name[5:7])
                img_label = int(img_name[8:11])
                category_label = ((object_label - 1) // 5) + 1
                img_dir = '/content/core50_128x128/' + ses + '/' + obj + '/' + img_name

                filewriter.writerow([img_dir, session_label, object_label, category_label, img_label])
