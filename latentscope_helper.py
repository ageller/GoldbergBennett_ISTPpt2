import os
import sys
import glob
import re
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import json

# for working in  jupyter notebook to disable the output from latentscope
from IPython.utils.capture import capture_output
  
import latentscope as ls

# latentscope produces a lot of output.  Sometimes I may want to supress it.  This function will allow me to do that.
import contextlib

@contextlib.contextmanager
def conditional_suppress_output(suppress):
    if suppress:
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            mplbackend = matplotlib.rcParams['backend']
            
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                matplotlib.use('AGG')

                if 'IPython' in sys.modules:
                    with capture_output() as captured:
                        yield
                else:
                    yield

                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                matplotlib.use(mplbackend)
    else:
        yield
class latentscope_helper(object):
    '''
    a class with helper methods for running latent-scope which I think makes this process more streamlined (and adaptable to different files)
    '''
    
    def __init__(self, 
        latent_scope_dir = None, # directory where the latentscope files are stored
        dataset_id = None, # data set name, for the sub-directory name within latent_scope_dir for this project
        data = None, # pandas DataFrame that contains the data to analyze
        text_column = None, # response column name from data_file
        embedding_number = None, # set the file number of embedding (if None, the next number in sequence is used)
        umap_number = None, # set the file number of umap (if None, the next number in sequence is used)
        cluster_number = None, # set the file number of cluster (if None, the next number in sequence is used)
        label_number = None, # set the file number of label (if None, the next number in sequence is used)
        scope_number = None, # set the file number of scope (if None, the next number in sequence is used)
        remove_old_files = False, # set this to True if you want to clean the latent-scope directories and start fresh
        quick_clean = False, # set this to True if you want to remove every file in the directories, regardless of imin and imax
        imin = 0, # minimum number of for files to search through (to remove)
        imax = 50, # maximum number of for files to search through (to remove)
        run_embedding = True, # whether to run the embedding step (and potentially remove previous files)
        run_umap = True, # whether to run the umap step (and potentially remove previous files)
        run_cluster = True, # whether to run the clustering step (and potentially remove previous files)
        run_label = True, # whether to run the labeling step (and potentially remove previous files)
        save_scope = True, # whether to save the scope to a file (after analysis)
        embedding_model_id = "transformers-jinaai___jina-embeddings-v2-small-en", # embeddings model name
        embedding_n_dimensions = 512, # number of dimensions for embedding.  reading the jina docs, they often use this number as an example for the number of dimensions (not sure this is a recommendation though)
        umap_n_components = 2, # number of UMAP dimensions
        umap_n_neighbors = 10, # "controls how UMAP balances local versus global structure in the data." Larger values mean UMAP will look at larger distances for neighbors (15 is default)
        umap_min_dist = 0, # "controls how tightly UMAP is allowed to pack points together" (default is 0.1)
        cluster_samples = 5, # min_cluster_size in HDBSCAN : "the smallest size grouping that you wish to consider a cluster"
        cluster_min_samples = 15, # min_samples in HDBSCAN : "provide a measure of how conservative you want your clustering to be. The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise, and clusters will be restricted to progressively more dense areas."
        cluster_selection_epsilon =  0.05, # cluster_selection_epsilon in HDBSCAN : distance measure between clusters that "ensures that clusters below the given threshold are not split up any further"
        chat_model_id = "transformers-TinyLlama___TinyLlama-1.1B-Chat-v1.0", # LLM to use for labeling the clusters
        label_length = 10, # max length to tell the LLM to use when generating a given label (not always respected by the LLM!)
        chat_model_instructions_before = "Below is a list of items each starting with [item].  Each item is a response from a different person to a survey. These items all have a similar theme.  The list begins below.", # string of text to provide the LLM as instructions before the list of cluster items is given
        chat_model_instructions_after = "That was the last item in the list.  Now return a concise label for the items in this list that describes the theme.  This label should not be fully verbatim text from any individual item.  Your label should contain no more than 10 words.", # string of text to provide the LLM as instructions after the list of cluster items is given
        scope_description = "Unlabeled scope", # label to give the scope (when using the latentscope server to view the data)
        suppress_latentscope_output = False, # if True all screen output from latentscope will be supressed
        suppress_helper_output = False, # if True all output I generated will be supressed
        suppress_all_output =  False, # if True this will result in no screen output 
        **kwargs):
    
        # required inputs
        self.latent_scope_dir = latent_scope_dir 
        self.dataset_id = dataset_id 
        self.data = data
        self.text_column = text_column

        # parameters defined by defaults, but that the user could set on input
        self.embedding_number = embedding_number 
        self.umap_number = umap_number 
        self.cluster_number = cluster_number 
        self.label_number = label_number 
        self.scope_number = scope_number 
        self.remove_old_files = remove_old_files
        self.quick_clean = quick_clean
        self.imin = imin
        self.imax = imax
        self.run_embedding = run_embedding
        self.run_umap = run_umap
        self.run_cluster = run_cluster
        self.run_label = run_label
        self.save_scope = save_scope
        self.embedding_model_id = embedding_model_id
        self.embedding_n_dimensions = embedding_n_dimensions
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.cluster_samples = cluster_samples 
        self.cluster_min_samples = cluster_min_samples 
        self.cluster_selection_epsilon = cluster_selection_epsilon 
        self.chat_model_id = chat_model_id
        self.label_length = label_length
        self.chat_model_instructions_before = chat_model_instructions_before
        self.chat_model_instructions_after = chat_model_instructions_after
        self.scope_description = scope_description
        self.suppress_latentscope_output = suppress_latentscope_output
        self.suppress_all_output = suppress_all_output
        self.suppress_helper_output = suppress_helper_output
        if (self.suppress_all_output):
            self.suppress_latentscope_output = True
            self.suppress_helper_output = True

        # internal file numbering system that will be set in initialize_latentscope
        self.umap_embedding_id = None
        self.cluster_umap_id = None
        self.label_cluster_id = None
        self.scope_label = None
        self.scope_labels_id = None

    def initialize_files_and_numbering(self, dirs_to_remove = None):
        # clean the latenscope directory and set the scope number

        def get_num(subdir, fileprefix):
            nums = []
            fls = []
            n_out = '00001'
            for f in glob.glob(os.path.join(self.latent_scope_dir, self.dataset_id, subdir, fileprefix + '-[0-9][0-9][0-9][0-9][0-9].json')):
                fls.append(f)
                f_split = os.path.split(f)[-1].replace(fileprefix, '')
                x = re.split('-|\.', f_split)
                nums.append(int(x[1]))
            if (len(nums) > 0):
                n = max(nums)
                n_out = str(n + 1).zfill(5)
            
            return n_out

        # REMOVE PREVIOUS FILES?... BEWARE
        if (self.remove_old_files):
            if (not self.suppress_helper_output):
                print('\nREMOVING OLD FILES ...\n')
            self.scope_number = self.embedding_number = self.umap_number = self.cluster_number = self.label_number = '00001'
            if (dirs_to_remove is None):
                dirs_to_remove = ['scopes']
                if (self.run_embedding):
                    dirs_to_remove.append('embeddings')
                if (self.run_umap):
                    dirs_to_remove.append('umaps')
                if (self.run_cluster):
                    dirs_to_remove.append('clusters')
            for d in dirs_to_remove:
                if (self.quick_clean):
                    # remove all the files regardless of number
                    for f in glob.glob(os.path.join(self.latent_scope_dir, self.dataset_id, d, '*')):
                        if (not self.suppress_helper_output):
                            print("removing : ",f)
                        os.remove(f)
                else:
                    # check by number
                    for i in range(self.imin, self.imax, 1):
                        for f in glob.glob(os.path.join(self.latent_scope_dir, self.dataset_id, d, '*'+str(i).zfill(5)+'*')):
                            if (not self.suppress_helper_output):
                                print("removing : ",f)
                            os.remove(f)
        else:

            if (self.embedding_number is None):
               self.embedding_number = get_num('embeddings', 'embedding')
            if (self.umap_number is None):
                self.umap_number = get_num('umaps', 'umap')
            if (self.cluster_number is None):
                self.cluster_number = get_num('clusters', 'cluster')
            if (self.label_number is None):
                self.label_number = get_num('clusters', 'cluster-' + str(min(int(self.cluster_number) - 1,1)).zfill(5) + '-labels')
            if (self.scope_number is None):
                self.scope_number = get_num('scopes', 'scope')
                # old code to get the next number from the scope file
                # with open(os.path.join(self.latent_scope_dir, self.dataset_id, "scopes", "scope-" + self.scope_number + ".json")) as jdata:
                #     scopes_info = json.load(jdata)
                #     self.embedding_number = scopes_info['embedding_id'].replace('embedding-','')
                #     self.umap_number = scopes_info['umap_id'].replace('umap-','')
                #     self.cluster_number = scopes_info['cluster_id'].replace('cluster-','')
                #     self.label_number = scopes_info['cluster_labels_id'].replace('cluster-' + self.cluster_number + '-labels-','')
            if (not self.suppress_helper_output):
                print('new embedding number = ', self.embedding_number)
                print('new umap number = ', self.umap_number)
                print('new cluster number = ', self.cluster_number)
                print('new label number = ', self.label_number)
                print('new scope number = ', self.scope_number)


    def initialize_latentscope_filenames(self):
        self.umap_embedding_id = "embedding-" + self.embedding_number
        self.cluster_umap_id = "umap-" + self.umap_number
        self.label_cluster_id = "cluster-" + self.cluster_number
        self.scope_labels_id = self.label_cluster_id + "-labels-" + self.label_number
        self.scope_label = "scope-" + self.scope_number

    def initialize_latentscope(self):

        if (not self.suppress_helper_output):
            print('\nINITIALIZING LATENTSCOPE ...\n')

        self.initialize_latentscope_filenames()

        with conditional_suppress_output(self.suppress_latentscope_output):
            # initialize latent-scope
            ls.init(self.latent_scope_dir)

            # ingest the data into latent-scope
            ls.ingest(self.dataset_id, self.data, text_column = self.text_column)


    def run_latentscope(self):

        # calculate the embeddings
        # dataset_id, text_column, model_id, prefix, rerun, dimensions
        # NOTE: the example notebook online did not have rerun or dimensions.  I looked at the code, and I think rerun should be None
        #       dimensions from the flask server = 384 (not sure where this number comes from!)
        if (self.run_embedding):
            if (not self.suppress_helper_output):
                print('\nCALCULATING EMBEDDINGS ...\n')
            with conditional_suppress_output(self.suppress_latentscope_output):
                ls.embed(self.dataset_id, self.text_column, self.embedding_model_id, "", None, self.embedding_n_dimensions, next_embedding_number = int(self.embedding_number))

        
        # run UMAP dimension reduction
        # dataset_id, embedding_id, n_neighbors, min_dist
        # NOTE: I added the n_components arg
        if (self.run_umap):
            if (not self.suppress_helper_output):
                print('\nRUNNING UMAP ...\n')
            with conditional_suppress_output(self.suppress_latentscope_output):
                ls.umap(self.dataset_id, self.umap_embedding_id, self.umap_n_neighbors, self.umap_min_dist, n_components = self.umap_n_components, next_umap_number = int(self.umap_number))

        # run HDBSCAN to cluster (on UMAP vectors)
        # dataset_id, umap_id, samples, min_samples
        # NOTE: the example from latent-scope's GitHub repo is missing an argument for "cluster_selection_epsilon"... 
        if (self.run_cluster):
            if (not self.suppress_helper_output):
                print('\nIDENTIFYING CLUSTERS ...\n')
            with conditional_suppress_output(self.suppress_latentscope_output):
                ls.cluster(self.dataset_id, self.cluster_umap_id, self.cluster_samples, self.cluster_min_samples, self.cluster_selection_epsilon, next_cluster_number = int(self.cluster_number))

        # run the LLM labeler
        # dataset_id, text_column, cluster_id, model_id, unused, rerun, instructions_before, instructions_after, label_length
        # NOTE: the code from GitHub was outdated and needed the last arg : rerun = None (or a value that points to a label), I added label_legth
        if (self.run_label):
            if (not self.suppress_helper_output):
                print('\nLABELLING CLUSTERS ...\n')
            with conditional_suppress_output(self.suppress_latentscope_output):
                ls.label(self.dataset_id, self.text_column, self.label_cluster_id, self.chat_model_id, "", None, self.chat_model_instructions_before, self.chat_model_instructions_after,  self.label_length, next_label_number = int(self.label_number))

        # save the scope
        # dataset_id, embedding_id, umap_id, cluster_id, labels_id, label, description
        if (self.save_scope):
            if (not self.suppress_helper_output):
                print('\nSAVING SCOPE ...\n')
            with conditional_suppress_output(self.suppress_latentscope_output):
                ls.scope(self.dataset_id, self.umap_embedding_id, self.cluster_umap_id, self.label_cluster_id, self.scope_labels_id, self.scope_label, self.scope_description, scope_id = self.scope_label)


    def print_labels(self):
        # print out the labels
        labels = pd.read_parquet(os.path.join(self.latent_scope_dir, self.dataset_id, "clusters", self.scope_labels_id + ".parquet"))
        if (not self.suppress_helper_output):
            for i, x in enumerate(labels['label'].to_list()):
                print(i, x)
        


    def create_bar_chart(self, filename = None):
        # get the labels
        labels = pd.read_parquet(os.path.join(self.latent_scope_dir, self.dataset_id, "clusters", self.scope_labels_id + ".parquet"))

        # match the indices from labels to the original data IDs and count the number of unique entries
        labels_list = []
        labels_num = []
        # print("cluster label, number of elements in cluster, number of unique student IDs associated with cluster")
        for index, row in labels.iterrows():
            labels_list.append(row['label'])
            labels_num.append(len(self.data.iloc[row['indices']]['ID'].unique()))
            # print(labels_list[-1], len(row['indices']), labels_num[-1])
        labels_frac = np.array(labels_num)/len(self.data)

        # sort (by creating a DataFrame)
        df = pd.DataFrame()
        df['label'] = labels_list
        df['frac'] = labels_frac
        df['num'] = labels_num
        df.sort_values(by = 'num', inplace = True, ascending = False)

        # create the figure and save it
        f, ax = plt.subplots(figsize = (10,10))
        y_pos = np.arange(len(df['label']))
        ax.barh(y_pos, df['frac'], align = 'center')
        ax.set_yticks(y_pos, labels = df['label'])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Fraction of responses including the given theme')
        ax.set_title('Themes from survey responses')

        if (filename is not None):
            f.savefig(filename, bbox_inches = 'tight')

        return f, ax
    
    def create_excel_workbook(self, data_raw, filename):
        # get the labels and read in the original data
        labels = pd.read_parquet(os.path.join(self.latent_scope_dir, self.dataset_id, "clusters", self.scope_labels_id + ".parquet"))

        # create a clean DataFrame for the labels mapping to the sheet name
        labels_sheet = pd.DataFrame()
        labels_sheet['label'] = labels['label']
        labels_sheet['sheet'] = [f'cluster{i + 1}' for i in labels.index]

        # create an Excel file and add all the sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            data_raw.to_excel(writer, sheet_name = 'input_data', index = False)
            labels_sheet.to_excel(writer, sheet_name = 'labels_map', index = False)

            # match the indices from labels to the original data IDs and create the new DataFrames
            for index, row in labels.iterrows():
                cl = self.data.iloc[row['indices']].rename(columns = {self.text_column : row['label']})
                # save to sheet in Excel
                cl.to_excel(writer, sheet_name = f'cluster{index + 1}', index=False)

    def calculate_metrics(self, embedding_number = None, umap_number = None, cluster_number = None, calc_inertia = True, calc_silhouette_coefficient = True, calc_calinski_harabasz_index = True, calc_davies_bouldin_index = True):
        # returns inertia, Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index
        # - a lower inertia value is generally better
        # - a higher Silhouette Coefficient score relates to a model with better defined clusters. 
        # - a higher Calinski-Harabasz score relates to a model with better defined clusters.
        # - a lower Davies-Bouldin index relates to a model with better separation between the clusters.

        if (embedding_number is None):
            embedding_number = self.embedding_number

        if (umap_number is None):
            umap_number = self.umap_number

        if (cluster_number is None):
            cluster_number = self.cluster_number

        # get the embeddings info
        embedding_file_root = os.path.join(self.latent_scope_dir, self.dataset_id, "umaps", "umap-" + embedding_number)
        with open(embedding_file_root + ".json") as jdata:
            embedding_info = json.load(jdata)
        X = pd.read_parquet(embedding_file_root + ".parquet").to_numpy()


        # get the umap info
        umap_file_root = os.path.join(self.latent_scope_dir, self.dataset_id, "umaps", "umap-" + umap_number)
        with open(umap_file_root + ".json") as jdata:
            umap_info = json.load(jdata)

        # get the cluster info
        cluster_file_root = os.path.join(self.latent_scope_dir, self.dataset_id, "clusters", "cluster-" + cluster_number)
        with open(cluster_file_root + ".json") as jdata:
            cluster_info = json.load(jdata)
        # load the labels
        labels = pd.read_parquet(cluster_file_root+ ".parquet")['cluster'].to_numpy(dtype='int32')
        
        inertia = sc = chi = dbi = None
        
        if (calc_inertia):
            # calculate the intertia
            unique_labels = np.unique(labels)
            inertia = 0
            for label in unique_labels:
                # Get the points in the current cluster
                cluster_points = X[labels == label]
                # Calculate the cluster center
                cluster_center = cluster_points.mean(axis = 0)
                # Sum of squared distances to the center
                inertia += ((cluster_points - cluster_center)**2).sum()

        if (calc_silhouette_coefficient):
            # calcuate the silhouette coefficient
            sc = metrics.silhouette_score(X, labels, metric='euclidean')

        if (calc_calinski_harabasz_index):
            # calculate the Calinski-Harabasz Index
            chi = metrics.calinski_harabasz_score(X, labels)
        
        if (calc_davies_bouldin_index):
            # calculate the Davies-Bouldin Index
            dbi = metrics.davies_bouldin_score(X, labels)

        return {'inertia':inertia, 'silhouette_coefficient':sc, 'calinski_harabasz_index':chi, 'davies_bouldin_index':dbi, 
                'embedding_info':embedding_info, 'umap_info':umap_info, 'cluster_info':cluster_info}
