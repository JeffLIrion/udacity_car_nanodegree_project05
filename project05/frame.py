import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from scipy.ndimage.measurements import label

#import PIL
#from PIL import ImageFont
#from PIL import Image
#from PIL import ImageDraw

#import image






def convert_color(img, cspace=None, cspace0='BGR'):
    """Convert ``img`` to colorspace ``cspace``
    
    Parameters
    ----------
    img : numpy.ndarray
        The image
    cspace : str, None
        The image will be converted to this colorspace (e.g., 'BGR')
    cspace0 : str
        The colorspace that ``img`` is currently in
    
    Returns
    -------
    numpy.ndarray
        The image in the specified colorspace
    
    """
    if cspace is not None and cspace != cspace0:
        return eval('cv2.cvtColor(img, cv2.COLOR_{0}2{1})'.format(cspace0, cspace))
    else:
        return img


# ======================================== #
#                                          #
#              `Frame` class               #
#                                          #
# ======================================== #


class Frame(object):
    def __init__(self, img=None, imgpath=None, min_row=0, max_row=None, scale=1., spatial_cspace='BGR', hist_cspace='BGR', hog_cspace='BGR',
                 spatial_size=32, hist_bins=32, hist_range=(0, 256), hog_orientations=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_cells_per_step=2,
                 hog_block_norm='L1', hog_channel='ALL', spatial_feat=True, hist_feat=True, hog_feat=True, heat_thresh=2, prev=None, prev_thresh=2):
        """Initialize a ``Frame`` object
        
        Parameters
        ----------
        img : numpy.ndarray, None
            The image from which we are finding cars
        imgpath : str, None
            The path to the ``img`` image
        min_row : int
            We will search for cars in the image between ``min_row`` and ``max_row``
        max_row : int, None
            We will search for cars in the image between ``min_row`` and ``max_row``
        scale : float
            The image will be rescaled by 1/``scale``
        spatial_cspace : str
            The colorspace used for extracting spatial color features
        hist_cspace : str
            The colorspace used for extracting color histogram features
        hog_cspace : str
            The colorspace used for extracting HOG features
        spatial_size : int
            For `bin_spatial`, the image will be resized to ``(spatial_size, spatial_size)``
        hist_bins : int
            The number of bins for the histogram in `color_hist`
        hist_range : tuple, list
            The lower and upper range of the bins in `color_hist`
        hog_orientations : int
            Number of orientation bins for `get_hog_features`
        hog_pix_per_cell : int
            The size of a cell is ``(pix_per_cell, pix_per_cell)`` for `get_hog_features`
        hog_cell_per_block : int
            There are ``(cell_per_block, cell_per_block)`` cells in each block for `get_hog_features`
        hog_cells_per_step : int
            Each HOG window is translated by ``hog_cells_per_step`` cells
        hog_block_norm : str {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}
            Block normalization method for `get_hog_features`
        hog_channel : int, 'ALL'
            The image channel for which we are getting HOG features for `get_hog_features`
        spatial_feat : bool
            Use spatial color features
        hist_feat : bool
            Use color histogram features
        hog_feat : bool
            Use HOG features
        heat_thresh : int
            Entries in the ``self.heatmap`` that are less than ``heat_thresh`` are deemed noise
        prev : list, None
            A list of the last `Frame` objects (`self.prev[0]` is the one immediately before)
        prev_thresh : int
            A threshold value when using previous vehicle detections
            
        """
        assert img is not None or imgpath is not None, "`img` or `imgpath` must be specified"
        
        self._img = img
        self.imgpath = imgpath
        
        self.scale = scale
        
        # colorspace parameters
        self.spatial_cspace = spatial_cspace
        self.hist_cspace = hist_cspace
        self.hog_cspace = hog_cspace
        
        # image size parameters
        self.max_row = max_row if max_row is not None else self.img0.shape[0]
        self.min_row = min_row
        
        # spatial color parameters
        self.spatial_size = spatial_size
        
        # color histogram parameters
        self.hist_bins = hist_bins
        self.hist_range = hist_range
        
        # HOG parameters
        self.hog_orientations = hog_orientations
        self.hog_pix_per_cell = hog_pix_per_cell
        self.hog_cell_per_block = hog_cell_per_block
        self.hog_cells_per_step = hog_cells_per_step
        self.hog_block_norm = hog_block_norm
        self.hog_channel = hog_channel
        
        # which features should be used?
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        
        # post-processing
        self.car_bboxes = None
        self.heatmap = None
        self.heat_thresh = heat_thresh
        self.prev = prev
        self.prev_thresh = prev_thresh
        
    def __str__(self):
        """Return a nicely formatted string containing all the parameters for the ``Frame``
        
        """
        description = '`Frame` object'
        description += '\n' + '='*len(description) + '\n\n'
        
        # image size
        rows0, cols0 = self.img0.shape[:2]
        if self.min_row == 0 and self.max_row == rows0 and self.scale == 1.:
            description += 'The image is {0} x {1}\n\n\n'.format(cols0, rows0)
        else:
            description += 'The original image is {0} x {1}\n\n'.format(cols0, rows0)
            
            if self.min_row != 0 or self.max_row != rows0:
                description += 'We only analyze rows {0} to {1}\n'.format(self.min_row, self.max_row)
            if self.scale != 1.:
                description += 'We downsize the image by a factor of {0}\n'.format(self.scale)
                
            rows, cols = self.img.shape[:2]
            description += 'The new image is {0} x {1}\n\n\n'.format(cols, rows)
        
        if self.spatial_feat:
            description += 'Spatial color parameters\n'
            description += '------------------------\n'
            description += 'Colorspace for spatial color features:  {0}\n'.format(self.spatial_cspace)
            description += 'Resized image size:  ({0}, {0})\n\n\n'.format(self.spatial_size)
        else:
            description += 'Spatial color features not used\n\n\n'
            
        if self.hist_feat:
            description += 'Color histogram parameters\n'
            description += '--------------------------\n'
            description += 'Colorspace for color histogram features:  {0}\n'.format(self.hist_cspace)
            description += 'Number of histogram bins: {0}\n'.format(self.hist_bins)
            description += 'Range of histogram bins:  ({0}, {1})\n\n\n'.format(self.hist_range[0], self.hist_range[1])
        else:
            description += 'Color histogram features not used\n\n\n'
            
        if self.hog_feat:
            description += 'HOG parameters\n'
            description += '--------------\n'
            description += 'Colorspace for HOG features:  {0}\n'.format(self.hog_cspace)
            description += 'Number of orientations:  {0}\n'.format(self.hog_orientations)
            description += 'Pixels per cell:  ({0}, {0})\n'.format(self.hog_pix_per_cell)
            description += 'Cells per block:  ({0}, {0})\n'.format(self.hog_cell_per_block)
            description += 'Cells per step:  {0}\n'.format(self.hog_cells_per_step)
            description += 'Block normalization method:  {0}\n'.format(self.hog_block_norm)
            description += 'HOG channel(s) used:  {0}\n\n\n'.format(self.hog_channel)
        else:
            description += 'HOG features not used\n\n\n'
            
        #description += 'Post-processing parameters\n'
        #description += '--------------------------\n'
        #description += 'Previous heatmaps used for car detection:  {0}\n'.format('None' if self.prev is None else len(self.prev))
        #description += 'Heatmap threshold:  {0}\n'.format(self.heat_thresh)
            
        return description
    
    def load_img(self):
        """Load a ``Frame`` object's associated image
        
        """
        if self.imgpath is not None:
            self._img = self.img0
            
    @property
    def img0(self):
        """Return a ``Frame`` object's original associated image
        
        Returns
        -------
        numpy.ndarray
            The ``Frame`` object's original image
        
        """
        if self._img is not None:
            return self._img
        else:
            return cv2.imread(self.imgpath)
        
    @property
    def img(self):
        """Return a ``Frame`` object's rescaled associated image
        
        Returns
        -------
        numpy.ndarray
            The ``Frame`` object's rescaled image
        
        """
        if self.scale != 1.:
            img0 = self.img0[self.min_row:self.max_row, :, :]
            rows, cols = img0.shape[:2]
            return cv2.resize(img0, (np.int(cols/self.scale), np.int(rows/self.scale)))
        else:
            return self.img0[self.min_row:self.max_row, :, :]
                
    def draw_bboxes(self, bbox_list, color=(0, 0, 255), thickness=6):
        """Draw boxes on the ``Frame`` object's image
        
        Parameters
        ----------
        bbox_list : list
            A list of ``((x1, y1), (x2, y2))`` coordinates for drawing boxes
        color : tuple
            The color that the boxes should be drawn
        thickness : int
            Thickness of lines that make up the rectangle
            
        Returns
        -------
        draw_img : numpy.ndarray
            The image with boxes drawn on it
        
        """
        draw_img = self.img0

        # draw each bounding box on your image copy using cv2.rectangle(
        for ((x1, y1), (x2, y2)) in bbox_list:
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)

        return draw_img

    # ======================================== #
    #                                          #
    #        Single feature extraction         #
    #                                          #
    # ======================================== #

    def bin_spatial(self, img, cspace0='BGR', visualize=False):
        """Compute binned color features
        
        Parameters
        ----------
        img : numpy.ndarray
            The image for which we are computing features
        cspace0 : str
            The colorspace that ``img`` is currently in (e.g., 'BGR')
        visualize : bool
            Also return an image of the spatial color feature
        
        Returns
        -------
        numpy.ndarray
            The stacked color features
        img : numpy.ndarray (if visualise=True)
            The color features image
        
        """
        img = convert_color(img, self.spatial_cspace, cspace0)
        
        if not visualize:
            return cv2.resize(img, (self.spatial_size, self.spatial_size)).ravel()
        else:
            return cv2.resize(img, (self.spatial_size, self.spatial_size)).ravel(), img


    def color_hist(self, img, cspace0='BGR', visualize=False):
        """Compute color histogram features
        
        Parameters
        ----------
        img : numpy.ndarray
            The image for which we are computing features
        cspace0 : str
            The colorspace that ``img`` is currently in
        visualize : bool
            Also return an image of the color histogram feature
        
        Returns
        -------
        numpy.ndarray
            The concatenated histograms
        hists : list (if visualise=True)
            A list of the histograms
            
        """
        img = convert_color(img, self.hist_cspace, cspace0)
            
        # Compute the histogram of the color channels separately
        if not visualize:
            return np.concatenate([np.histogram(img[:,:,i], bins=self.hist_bins, range=self.hist_range)[0] for i in range(3)])
        else:
            hists = [np.histogram(img[:,:,i], bins=self.hist_bins, range=self.hist_range)[0] for i in range(3)]
            return np.concatenate(hists), hists


    def get_hog_features(self, img, cspace0='BGR', visualize=False, feature_vector=False):
        """Extract a Histogram of Oriented Gradients (HOG) for the ``Frame`` object's image
        
        Parameters
        ----------
        img : numpy.ndarray
            The image for which we are computing features
        cspace0 : str
            The colorspace that ``img`` is currently in (e.g., 'BGR')
        visualize : bool
            Also return an image of the HOG
        feature_vector : bool
            Return the data as a feature vector by calling .ravel() on the result
            just before returning
            
        Returns
        -------
        features : list
            A list of HOG arrays (1 numpy array for each image channel)
        hog_image : list (if visualise=True)
            A list of visualisations of the HOG images (1 numpy array for each image channel)
        
        """
        img = convert_color(img, self.hog_cspace, cspace0)
        
        # all image channels
        if self.hog_channel == 'ALL':
            num_channels = img.shape[2]
            features = [hog(img[:,:,i], orientations=self.hog_orientations,
                            pixels_per_cell=(self.hog_pix_per_cell, self.hog_pix_per_cell),
                            cells_per_block=(self.hog_cell_per_block, self.hog_cell_per_block),
                            block_norm=self.hog_block_norm, transform_sqrt=False,
                            visualise=visualize, feature_vector=feature_vector)
                        for i in range(num_channels)]
                        
            if visualize:
                return list(zip(*features))
            else:
                return features
                
        # 1 image channel
        else:
            features = hog(img[:,:,hog_channel], orientations=self.hog_orientations,
                           pixels_per_cell=(self.hog_pix_per_cell, self.hog_pix_per_cell),
                           cells_per_block=(self.hog_cell_per_block, self.hog_cell_per_block),
                           block_norm=self.hog_block_norm, transform_sqrt=False,
                           visualise=visualize, feature_vector=feature_vector)
                           
            if visualize:
                return [features[0]], [features[1]]
            else:
                return [features]
        
    # ======================================== #
    #                                          #
    #            Feature extraction            #
    #                                          #
    # ======================================== #
    
    def extract_features(self):
        """Extract HOG, spatial and/or color features from an image
        
        This method is for extracting features from training images
        
        See `Frame.bin_spatial`, `Frame.color_hist`, and `Frame.get_hog_features`
        
        Returns
        -------
        features : numpy.ndarray
            The concatenated features
        
        """
        img = self.img0
        
        # Create a list to append feature vectors to
        features = []
        
        if self.spatial_feat:
            features.append(self.bin_spatial(img, 'BGR'))
            
        if self.hist_feat:
            features.append(self.color_hist(img, 'BGR'))

        if self.hog_feat:
            features.append(np.concatenate(self.get_hog_features(img, 'BGR', False, True)))
        
        # Return list of feature vectors
        return np.concatenate(features)

    # ======================================== #
    #                                          #
    #                Find cars                 #
    #                                          #
    # ======================================== #
    
    def get_windows(self):
        """Get a list of all the windows that will be searched for cars
        
        Returns
        -------
        bbox_list : list
            A list with entries of the form ``(xy, xy_hog, xy_color)``, where each
            ``xy`` term is of the form ``((x0, y0), (x1, y1))`` and specifies a
            bounding box for the original image, the HOG-processed image, and the
            patch image used by `Frame.color_hist` and `Frame.bin_spatial`
        
        """
        rows, cols = self.img.shape[:2]
        
        # Define blocks and steps
        nxblocks = (cols // self.hog_pix_per_cell) - self.hog_cell_per_block + 1
        nyblocks = (rows // self.hog_pix_per_cell) - self.hog_cell_per_block + 1
        nfeat_per_block = self.hog_orientations * self.hog_cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.hog_pix_per_cell) - self.hog_cell_per_block + 1
        nxsteps = (nxblocks - nblocks_per_window) / self.hog_cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // self.hog_cells_per_step
        
        bbox_list = []
        for xb in np.linspace(0, nxsteps, np.int(nxsteps)+1):
            for yb in range(nysteps+1):
                ypos = yb*self.hog_cells_per_step
                xpos = xb*self.hog_cells_per_step
                xleft = xpos * self.hog_pix_per_cell
                ytop = ypos * self.hog_pix_per_cell

                xbox_left = np.int(xleft * self.scale)
                ytop_draw = np.int(ytop * self.scale)
                win_draw = np.int(window * self.scale)
                  
                bbox_list.append((((xbox_left, ytop_draw + self.min_row), (xbox_left + win_draw, ytop_draw + win_draw + self.min_row)),
                                  ((np.int(xpos), ypos), (np.int(xpos) + nblocks_per_window, ypos + nblocks_per_window)),
                                  ((np.int(xleft), ytop), (np.int(xleft) + window, ytop + window))))
            
        return bbox_list
    
    def get_bboxes(self, svc=None, X_scaler=None, scales=[1], min_rows=[None], max_rows=[None], cells_per_steps=[None]):
        """Extract features from an image, apply the classifier, and generate ``self.heatmap``
        
        This method is for finding cars on a real world image (i.e., not a cropped training image)
        
        See `Frame.bin_spatial`, `Frame.color_hist`, and `Frame.get_hog_features`
        
        Parameters
        ----------
        svc : sklearn.svm.classes.LinearSVC
            SVM classifier
        X_scaler : sklearn.preprocessing.data.StandardScaler
            Feature scaler
        scales : list
            The scales (``self.scale``) at which to run the feature extraction and classifier
        min_rows : list
            A list of the minimum rows for cropping the image
        max_rows : list
            A list of the maximum rows for cropping the image
        cells_per_steps : list
            A list of the values for ``self.hog_cells_per_step``
        
        """
        bbox_list = []
        self.heatmap = np.zeros(self.img0.shape[:2], dtype=np.float)
        
        for scale, min_row, max_row, cells_per_step in zip(scales, min_rows, max_rows, cells_per_steps):
            self.scale = scale
            self.min_row = min_row if min_row is not None else self.img0.shape[0]//2
            self.max_row = max_row if max_row is not None else self.img0.shape[0]
            self.hog_cells_per_step = cells_per_step
            
            img = self.img

            # Compute a list of the individual channel HOG features for the entire image
            if self.hog_feat:
                hog_list = self.get_hog_features(img, 'BGR', False, False)
                
            for xy, xy_hog, xy_color in self.get_windows():
                # Extract HOG for this patch
                if self.hog_feat:
                    hog_features = np.hstack([hog_feat[xy_hog[0][1]:xy_hog[1][1], xy_hog[0][0]:xy_hog[1][0]].ravel() for hog_feat in hog_list])
                else:
                    hog_features = []
                    
                # Extract the image patch
                patch = cv2.resize(img[xy_color[0][1]:xy_color[1][1], xy_color[0][0]:xy_color[1][0]], (64,64))

                # Get spatial color features
                if self.spatial_feat:
                    spatial_features = self.bin_spatial(patch, 'BGR')
                else:
                    spatial_features = []
                    
                # get color histogram features
                if self.hist_feat:
                    hist_features = self.color_hist(patch, 'BGR')
                else:
                    hist_features = []
                    
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    bbox_list.append(xy)
                    
        return bbox_list
        
    def get_heatmap(self, **kwargs):
        """Extract features from an image, apply the classifier, and generate ``self.heatmap``
        
        See `Frame.get_bboxes`
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for `Frame.get_heatmap`
        
        This method is for finding cars on a real world image (i.e., not a cropped training image)
        
        See `Frame.bin_spatial`, `Frame.color_hist`, and `Frame.get_hog_features`
        
        Parameters
        ----------
        svc : sklearn.svm.classes.LinearSVC
            SVM classifier
        X_scaler : sklearn.preprocessing.data.StandardScaler
            Feature scaler
        scales : list
            The scales (``self.scale``) at which to run the feature extraction and classifier
        min_rows : list
            A list of the minimum rows for cropping the image
        max_rows : list
            A list of the maximum rows for cropping the image
        cells_per_steps : list
            A list of the values for ``self.hog_cells_per_step``
        
        """
        bbox_list = self.get_bboxes(**kwargs)
        
        self.heatmap = np.zeros(self.img0.shape[:2], dtype=np.float)
        
        for xy in bbox_list:
            self.heatmap[xy[0][1]:xy[1][1], xy[0][0]:xy[1][0]] += 1
        
    def find_cars(self, **kwargs):
        """Find cars using the heatmap
        
        See `Frame.get_heatmap`
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for `Frame.get_bboxes` / `Frame.get_heatmap`
            
        """
        if self.heatmap is None:
            self.get_heatmap(**kwargs)
            
        heatmap = np.copy(self.heatmap)
            
        if self.prev is None:
            # find the cars from the heatmap
            heatmap[heatmap < self.heat_thresh] = 0
            self.car_bboxes = heatmap_to_bboxes(heatmap)
        
            return self.draw_bboxes(self.car_bboxes)
            
        else:
            # use previous heatmaps
            for p in self.prev:
                heatmap += p.heatmap
            heatmap /= (1 + len(self.prev))
                
            heatmap = np.clip(heatmap, 0, 255)
            
            # find the cars from the heatmap
            heatmap[heatmap < self.heat_thresh] = 0
                
            self.car_bboxes = heatmap_to_bboxes(heatmap)
            
            heatmap2 = np.zeros(self.heatmap.shape[:2], dtype=np.float)
            
            # fill in each bounding box on the heatmap
            for ((x1, y1), (x2, y2)) in self.car_bboxes:
                heatmap2[y1:y2, x1:x2] += 1
                
            for p in self.prev:
                for ((x1, y1), (x2, y2)) in p.car_bboxes:
                    heatmap2[y1:y2, x1:x2] += 1
                    
            car_bboxes = heatmap_to_bboxes(heatmap2)
            
            return self.draw_bboxes(self.car_bboxes)
        
def heatmap_to_bboxes(heatmap):
    """Convert from a heatmap image to car bounding boxes
    
    Parameters
    ----------        
    heatmap : numpy.ndarray
        A heatmap image of car detections
    
    Returns
    -------
    bbox_list : list
        A list of ``((x1, y1), (x2, y2))`` coordinates for drawing boxes
    
    """
    bbox_list = []
    
    # convert from a heatmap to labels
    labels = label(heatmap)
    
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Define a bounding box based on min/max x and y
        bbox_list.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
        
    return bbox_list
