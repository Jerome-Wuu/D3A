from scipy.cluster.vq import *
from PIL import Image
from PIL import ImageOps
from pylab import *
import os
import utils
import random
from scipy.stats import mode
from arguments import parse_args
from scipy.spatial.distance import cdist


def kmeans(X, n_clusters, fixed_centers=None, max_iter=100):
    # Initialize centers
    if fixed_centers is None:
        centers = X[np.random.choice(len(X), n_clusters, replace=False)]
    else:
        centers = fixed_centers

    # Iterate until convergence or max iterations
    for i in range(max_iter):
        # Assign each point to nearest center
        labels = np.argmin(cdist(X, centers), axis=1)

        # Update centers
        for j in range(n_clusters):
            if fixed_centers is not None and np.array_equal(centers[j], fixed_centers[j]):
                continue
            centers[j] = np.mean(X[labels == j], axis=0)

    return labels, centers


def main(args):

    work_dir = os.path.join(args.log_dir, args.data_domain)
    JPEGImages_dir = os.path.join(work_dir, 'JPEGImages')
    SegmentationClass_dir = os.path.join(work_dir, 'SegmentationClass')
    # ImagesSets_dir = os.path.join(work_dir, 'ImagesSets')
    folder_path = '../DMCC_test/JPEGImages/'
    mask_path = '../DMCC_test/SegmentationClass/'
    SegmentationCode_dir = utils.make_dir(os.path.join(work_dir, 'SegmentationCode'))
    # code_path = '../DMCC_test/Code/'
    image_files = os.listdir(JPEGImages_dir)
    # selected_image = random.sample(image_files, 5)

    # for image_name in selected_image:
    for image_name in image_files:
        image_path = os.path.join(JPEGImages_dir, image_name)

        im = plt.imread(image_path)
        X = im.reshape(-1, 3)

        # Find two farthest pixels as initial centers
        distances = cdist(X, X)
        farthest_idx = np.unravel_index(np.argmax(distances), distances.shape)
        centers = X[farthest_idx[0]], X[farthest_idx[1]]

        # Perform kmeans with fixed centers
        labels, centers = kmeans(X, 2, fixed_centers=centers)

        # Reshape labels to image shape
        mode_val = mode(labels)[0][0]
        labels = where(mode_val, 1 - labels, labels)
        labels = labels.reshape(im.shape[:2])

        # Save segmented image
        np.savetxt('{}/{}.txt'.format(SegmentationCode_dir, image_name), labels, fmt='%d')
        labels = np.stack((labels, labels, labels), axis=-1) * 255
        file_name = "{}/{}.png".format(SegmentationClass_dir, image_name.split('.')[0])
        Image.fromarray(np.uint8(labels)).save(file_name)
    print('Completed img2mask for', work_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
