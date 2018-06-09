from sfd_detector import SFD_NET
import argparse
from os.path import abspath, dirname, join
import progressbar
import caffe

# Default models if none is passed to the __init__ method, assuming the `sfd_test_code` is in ${CAFFE_ROOT}/SFD/sfd_test_code
# These models are automatically downloaded by SFD/scripts/data/download_model.sh
MODEL_DEF = '{}/../../SFD_MODEL/deploy.prototxt'.format(abspath(dirname(__file__)))
MODEL_WEIGHTS = '{}/../../SFD_MODEL/SFD.caffemodel'.format(abspath(dirname(__file__)))

def process_imgs_list(imgs_list_file, output_file, dataset_path, origin, net):
    with open(imgs_list_file, 'r') as img_names:
        names = img_names.readlines()

    bar = progressbar.ProgressBar(max_value=len(names))
    with open(output_file, 'w') as f:
        for i, Name in enumerate(names):
            Image_Path = join(dataset_path, Name[:-1].replace('.jpg', '') + '.jpg')
            image = caffe.io.load_image(Image_Path)
        
            shrink = 1
            shrink = 640.0 / max(image.shape[0], image.shape[1])
        
            detections = net.detect(image, shrink=shrink)

            for det in detections:
                xmin, ymin, xmax, ymax, score = det
                # Simple fitting to AFW/PASCAL, because the gt box of training
                # data (i.e., WIDER FACE) is longer than the gt box of AFW/PASCAL
                ymin += 0.2 * (ymax - ymin + 1)   

                f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                        format(Name[:-1], score, xmin, ymin, xmax, ymax))

            bar.update(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test code for a SFD trained model.')
    parser.add_argument('-d', '--dataset', type=str, choices=['AFW', 'PASCAL'],
        help='Dataset name to test. Options: AFW, PASCAL', required=True)
    parser.add_argument('-p', '--path', type=str, help='Dataset path', required=True)
    parser.add_argument('--model', type=str, help='Path to Caffe prototxt', default=MODEL_DEF, required=False)
    parser.add_argument('--weights', type=str, help='Path to Caffe weights (caffemodel)', default=MODEL_WEIGHTS, required=False)
    parser.add_argument('--device', type=int, default=0, help="GPU device to use, default is 0")
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_path = args.path
    model = args.model
    weights = args.weights
    device = args.device

    imgs_list = 'output/{}/{}_img_list.txt'.format(dataset_name, dataset_name.lower())
    dets_file = 'output/{}/sfd_{}_dets.txt'.format(dataset_name, dataset_name.lower())
    net = SFD_NET(model_file=model, pretrained_file=weights, device=device)
    process_imgs_list(imgs_list, dets_file, dataset_path, dataset_name, net)
