from __future__ import print_function
import csv
import logging
import os
import re
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor


# run in Windows 10
nifti_path = r'..\..\QSM_inMNI'
roi_dir = r'.\..\remote_data\QSM\aal3v1_roi'
head = 'm'
params = './Params.yaml'
outPath = './radiomics_features'


def main():
    os.makedirs(outPath, exist_ok=True)
    maskFilepath = os.path.join(outPath, 'roi_mask_161to164_SN.nii.gz')
    outputFilepath = os.path.join(outPath, 'radiomics_features.csv')
    progress_filename = os.path.join(outPath, 'pyrad_log.txt')

    # Configure logging
    rLogger = logging.getLogger('radiomics')

    # Set logging level
    # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

    # Create handler for writing to log file
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')

    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)

    logger.info('pyradiomics version: %s', radiomics.__version__)

    flists = [kk for kk in os.listdir(nifti_path) if kk.startswith(head)]

    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    headers = None
    for idx, ff in enumerate(flists, start=1):

        logger.info("(%d/%d) Processing Patient (Image: %s)", idx, len(flists), ff)
        imageFilepath = os.path.join(nifti_path, ff)

        featureVector = dict()
        featureVector['ID'] = int(re.search('\d+', ff).group())
        featureVector['Image'] = ff
        featureVector['Mask'] = os.path.basename(maskFilepath)

        try:
            featureVector.update(extractor.execute(imageFilepath, maskFilepath, label=None))

            with open(outputFilepath, 'a') as outputFile:
                writer = csv.writer(outputFile, lineterminator='\n')
                if headers is None:
                    headers = list(featureVector.keys())
                    writer.writerow(headers)
                row = []
                for h in headers:
                    row.append(featureVector.get(h, "N/A"))
                writer.writerow(row)

        except Exception:
            logger.error('FEATURE EXTRACTION FAILED', exc_info=True)


if __name__ == '__main__':
    main()
