import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import latentscope as ls

class latentscope_helper(object):
    '''
    a class with helper methods for running latent-scope which I think makes this process more streamlined (and adaptable to different files)
    '''
    
    def __init__(self, 
        latent_scope_dir = None, # directory where the latentscope files are stored
        dataset_id = None, # data set name, for the sub-directory name within latent_scope_dir for this project
        data = None, # pandas DataFrame that contains the data to analyze
        text_column = None, # response column name from data_file
        scope_number = 'new', # number that will be appended to all latentscope files 
        remove_old_files = False, # set this to True if you want to clean the latent-scope directories and start fresh
        imax = 50, # maximum number of scopes that it should search through
        run_embedding = True, # whether to run the embedding step (and potentially remove previous files)
        run_umap = True, # whether to run the umap step (and potentially remove previous files)
        run_cluster = True, # whether to run the clustering step (and potentially remove previous files)
        run_label = True, # whether to run the labeling step (and potentially remove previous files)
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
        scope_description = "Unlabeled scope", #label to give the scope (when using the latentscope server to view the data)
        **kwargs):
    
        # required inputs
        self.latent_scope_dir = latent_scope_dir 
        self.dataset_id = dataset_id 
        self.data = data
        self.text_column = text_column

        # parameters defined by defaults, but that the user could set on input
        self.scope_number = scope_number 
        self.remove_old_files = remove_old_files
        self.imax = imax
        self.run_embedding = run_embedding
        self.run_umap = run_umap
        self.run_cluster = run_cluster
        self.run_label = run_label
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

        # internal file numbering system that will be set in initialize_latentscope
        self.umap_embedding_id = None
        self.cluster_umap_id = None
        self.label_cluster_id = None
        self.scope_label = None
        self.scope_labels_id = None

    def initialize_files_and_scope(self):
        # clean the latenscope directory and set the scope number

        # REMOVE PREVIOUS FILES?... BEWARE
        if (self.remove_old_files):
            print('\nREMOVING OLD FILES ...\n')
            self.scope_number = '001'
            dir_to_remove = ['scopes']
            if (self.run_embedding):
                dir_to_remove.append('embeddings')
            if (self.run_umap):
                dir_to_remove.append('umaps')
            if (self.run_cluster):
                dir_to_remove.append('clusters')
            for i in range(self.imax):
                for d in dir_to_remove:
                    for f in glob.glob(os.path.join(self.latent_scope_dir, self.dataset_id, d, '*'+str(i).zfill(3)+'*')):
                        print("removing : ",f)
                        os.remove(f)
        else:
            if (self.scope_number ==  'new'):
                nums = []
                fls = []
                for f in glob.glob(os.path.join(self.latent_scope_dir, self.dataset_id, 'embeddings', '*.json')):
                    fls.append(f)
                    x = re.split('-|\.', f)
                    nums.append(int(x[-2]))
                if (len(nums) > 0):
                    n = max(nums)
                    self.scope_number = str(n + 1).zfill(3)
                else:
                    self.scope_number = '001'
                print('list of files :', fls)
                print('list of numbers :', nums)
                print('new scope number = ', self.scope_number)


    def initialize_latentscope(self):

        print('\nINITIALIZING LATENTSCOPE ...\n')

        self.umap_embedding_id = "embedding-" + self.scope_number
        self.cluster_umap_id = "umap-" + self.scope_number
        self.label_cluster_id = "cluster-" + self.scope_number
        self.scope_labels_id = self.label_cluster_id + "-labels-" + self.scope_number
        self.scope_label = "Scope" + self.scope_number

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
            print('\nCALCULATING EMBEDDINGS ...\n')
            ls.embed(self.dataset_id, self.text_column, self.embedding_model_id, "", None, self.embedding_n_dimensions)

        
        # run UMAP dimension reduction
        # dataset_id, embedding_id, n_neighbors, min_dist
        # NOTE: I added the n_components arg
        if (self.run_umap):
            print('\nRUNNING UMAP ...\n')
            ls.umap(self.dataset_id, self.umap_embedding_id, self.umap_n_neighbors, self.umap_min_dist, n_components = self.umap_n_components)

        # run HDBSCAN to cluster (on UMAP vectors)
        # dataset_id, umap_id, samples, min_samples
        # NOTE: the example from latent-scope's GitHub repo is missing an argument for "cluster_selection_epsilon"... 
        if (self.run_cluster):
            print('\nIDENTIFYING CLUSTERS ...\n')
            ls.cluster(self.dataset_id, self.cluster_umap_id, self.cluster_samples, self.cluster_min_samples, self.cluster_selection_epsilon)

        # run the LLM labeler
        # dataset_id, text_column, cluster_id, model_id, unused, rerun, instructions_before, instructions_after, label_length
        # NOTE: the code from GitHub was outdated and needed the last arg : rerun = None (or a value that points to a label), I added label_legth
        if (self.run_label):
            print('\nLABELLING CLUSTERS ...\n')
            ls.label(self.dataset_id, self.text_column, self.label_cluster_id, self.chat_model_id, "", None, self.chat_model_instructions_before, self.chat_model_instructions_after,  self.label_length)

        # dataset_id, embedding_id, umap_id, cluster_id, labels_id, label, description
        print('\nSAVING SCOPE ...\n')
        ls.scope(self.dataset_id, self.umap_embedding_id, self.cluster_umap_id, self.label_cluster_id, self.scope_labels_id, self.scope_label, self.scope_description)


    def print_labels(self):
        # print out the labels
        labels = pd.read_parquet(os.path.join(ls.get_data_dir(), self.dataset_id, "clusters", self.scope_labels_id + ".parquet"))
        for i, x in enumerate(labels['label'].to_list()):
            print(i, x)
        


    def create_bar_chart(self, filename = None):
        # get the labels
        labels = pd.read_parquet(os.path.join(ls.get_data_dir(), self.dataset_id, "clusters", self.scope_labels_id + ".parquet"))

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
        labels = pd.read_parquet(os.path.join(ls.get_data_dir(), self.dataset_id, "clusters", self.scope_labels_id + ".parquet"))

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