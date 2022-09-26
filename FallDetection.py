import argparse
from utils import FeatureExtractor, KeyPoints

parser = argparse.ArgumentParser(description='Process a video. ')
parser.add_argument('-video', metavar='Video', help='The video to be processed',required=True)
parser.add_argument('-m', '--method', metavar = 'Method', help='The type of the cost calculated. Available methods: Division, MeanDifference, DifferenceMean, DifferenceSum, Mean', nargs='?', default='DifferenceMean', const='DifferenceMean')
parser.add_argument('--save', action=argparse.BooleanOptionalAction, help = 'Save or not save the image')

args = parser.parse_args()

if __name__ == '__main__':

	featureextractor = FeatureExtractor()
	cost = featureextractor.realTimeVideo(str(args.video), str(args.method), args.save)


#     featureextractor = FeatureExtractor()
#     costmethod = 'Division'
#     cost = featureextractor.processVideo('.//TestData//Fall3.mp4', costmethod)
#     featureextractor.separatePlot(cost, costmethod, save=True)