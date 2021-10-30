import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import collections

" set up pandas display options "
desired_width = 1000
desired_length = 1000
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', desired_width)
pd.set_option('display.max_rows', desired_length)


def calculate_similarity_matrix(compounds):

    number_compounds = len(compounds) # number of comparisons
    similarities = np.zeros((number_compounds,number_compounds)) # initializing Matrix for sim scoring


    # create similarity score matrix (half for efficiency) using RDkit
    for row in range(1,number_compounds+1):

        # Exception for case that given smile cant be digested by RDKit

        row_fps = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(compounds[row-1]),
                                                        useChirality=False, radius=2,nBits=500) # done here to do it only once (twice actually) per iteration

        for column in range(1,row+1):

            column_fps = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(compounds[column-1]),
                                                               useChirality=False, radius=2,nBits=500)
            similarities[row-1,column-1] = DataStructs.FingerprintSimilarity(row_fps, column_fps) #Tanimoto similarity

    return similarities

def clustered_heatmap(mat, compounds, title, method = 'single', metric = 'euclidean', thresh = 0.9, colored_ticks=None):



    mat[mat==1] = 0.5
    mat = mat + mat.T # turning half matrix into full symmetric matrix for scipy clustering

    g = sns.clustermap(mat,method=method,metric=metric) # create a cluster object
    resorted_similarities = g.data2d # get clustered sym. matrix as pd df with sorted compounds as header
    resorted_similarities = np.tril(resorted_similarities) # upper triangle of matrix removed (its symmetric) and
    # now contained in numpy array without headers

    resorted_compound_pointer = list(g.data2d.columns) # get only list of compound numbers (df headers in default order)

    resorted_compounds = []

    # resorting compound chembl id to match new order
    for pointer in resorted_compound_pointer:
        resorted_compounds.append(compounds[pointer])

    if colored_ticks:
        st = set(colored_ticks)
        color_label_pos = [i for i, e in enumerate(resorted_compounds) if e in st]

    Z = linkage(mat, method=method, metric=metric)
    #dn = dendrogram(Z)
    #fig = plt.figure(figsize=(25, 10))
    #plt.show()

    # Extracting clusters
    clus = fcluster(Z,t=thresh, criterion='distance')

    # Extract non single entry clusters
    cluster_pointer = [item for item, count in collections.Counter(clus).items() if count > 1]
    cluster_accumulation = [] # array of arrays storing the clusters


    # Define Figure for heatmap
    fig, ax = plt.subplots(figsize=(20,20))

    if len(resorted_compound_pointer) < 260:
        # Lable the axis with compound names
        ax.set_xticks(np.arange(len(resorted_compound_pointer)))
        ax.set_yticks(np.arange(len(resorted_compound_pointer)))
        ax.set_xticklabels(resorted_compounds, {'fontsize':12})
        ax.set_yticklabels(resorted_compounds, {'fontsize':12})

        if colored_ticks:
            for pos in color_label_pos:
                ax.get_xticklabels()[pos].set_color('red')
                ax.get_yticklabels()[pos].set_color('red')

        # Rotate x axis lables
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                 rotation_mode="anchor")

    else:
        # Lable the axis with compound names
        ax.set_xticks(np.arange(len(resorted_compound_pointer)))
        ax.set_yticks(np.arange(len(resorted_compound_pointer)))
        ax.set_xticklabels('')
        ax.set_yticklabels('')

    # Assign chembl ID's to their corresponding cluster
    for pointer in cluster_pointer:
        cluster_accumulation.append(compounds[clus==pointer])

    cluster_in_numbers = []

    for cluster in cluster_accumulation:
        cluster_numbers = []
        mapped_to_coordinates = []

        for chembl_id in cluster:
            current_cluster_numbers = np.where(compounds == '{}'.format(chembl_id))[0][0]
            cluster_numbers.append(current_cluster_numbers)
            # get positions mapped to heatmap coordinates for the current cluster
            ''' Little confusing part 
            the heatmap which is drawn from the clustered data has usual cartesian coordinated starting from 0. 
            Only lables are changed to the Reordered compound after clustering. This "Reordered" vector positions 
            needed to be translated back to the cartesian format used in the heatmap. 
            
            Process: Vector of Compounds 0 - N --> Reordering (cluster) --> Get coordinates by comparing single cluster 
            with reordered vector (like where (index) are compound a , b, d which are one cluster in the new vector)
            --> Translate into horizontal and vertical vector spanning the whole cluster --> Draw representive line in 
            heatmap '''
            line_range_current = np.where(np.isin(resorted_compound_pointer,current_cluster_numbers))
            mapped_to_coordinates.append(line_range_current)

        # get x and y points for vertical line
        # -+ 0.5 values are to draw lines enclosing the cluster. Else they would lay in middle of
        # a single heat map box
        y_vertical = [np.min(mapped_to_coordinates)-0.5, np.max(mapped_to_coordinates)+0.5]
        x_vertical = [np.min(mapped_to_coordinates)-0.5, np.min(mapped_to_coordinates)-0.5]

        # draw vertical line
        ax.plot(x_vertical,y_vertical, c='red', linewidth=3, markersize=12)

        # get x and y points for horizontal line
        y_horizontal = [np.max(mapped_to_coordinates)+0.5,np.max(mapped_to_coordinates)+0.5]
        x_horizontal = [np.min(mapped_to_coordinates)-0.5, np.max(mapped_to_coordinates)+0.5]

        # draw horizontal line
        ax.plot(x_horizontal,y_horizontal, c='red', linewidth=3, markersize=12)


        cluster_in_numbers.append(cluster_numbers)


    # create custom color map to block 0 values
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    white = [1,1,1,1]
    newcolors[0] = white
    new_cmp = ListedColormap(newcolors)
    ax.set_title(title,fontsize=30, weight=12)
    fig.tight_layout()


    #plot the heatmap
    im = ax.imshow(resorted_similarities, aspect='auto', cmap=new_cmp)
    #create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel('Tanimoto similarity', rotation=-90, va="bottom", fontsize=28, weight=15)
    plt.show()

    print('Number of clusters: {}'.format(len(cluster_accumulation)))



    return compounds, resorted_compounds, cluster_accumulation, g.data2d





def common_substructure(smile_list, thresh, timeout=60, compare_first=True):

    mols = []

    for smile in smile_list:
        mols.append(Chem.MolFromSmiles(smile))

    mol1 = mols[0]

    # compare for substructure
    # timeout: returns whatever it has after given amount of second
    # thresh: fraction of the dataset that must contain the MCS
    res = rdFMCS.FindMCS(mols, timeout=timeout, threshold=thresh)

    # get common substructure as smile
    smart_string = res.smartsString
    sub_structure = Chem.MolFromSmarts(smart_string) # extracting substructure

    if compare_first:
        # compare with just the first atom to get substructure (will work good if chemical similarity is assumed to be
        # high amoung all compounds --> Thresh should also set to be 1 then
        atom_indices = mol1.GetSubstructMatch(sub_structure)

        if atom_indices:
            common_smile = Chem.MolFragmentToSmiles(mol1, atom_indices)
        else:
            print(f'No substructure found for thresh: {thresh} and timeout: {timeout}')

            return None

    elif not compare_first:
        # compare with all to get fraction // computational intensive
        longest = 0
        common_smile = []
        for molecule in mols:
            atom_indices = molecule.GetSubstructMatch(sub_structure)
            if atom_indices:
                common_smile_current = Chem.MolFragmentToSmiles(molecule, atom_indices)

                # only advance atom indecies if they are getting longer
                if len(common_smile_current) > longest:
                    longest = len(common_smile_current)
                    common_smile = common_smile_current
                else:
                    continue
            else:
                continue



    return common_smile, smart_string

if __name__ == '__main__':
    # Importing data
    filepath = '/Users/mbpro/Desktop/CMS Master/Master Thesis/Clean_Project_Code/Data' \
               '/significant_p_valued_data_ecdf_O60885_BD1.csv'
    data = pd.read_csv(filepath)

    # Drop repeats of same compound in dataset
    data = data.drop_duplicates(subset='Smile').reset_index(drop=True)

    # Drop zero entries
    # .dropna is not used because entries are read in as strings (not optimal)
    data = data.query('Smile != "0"')

    # calculate all to all similarity matrix (should be stored locally at some point to not rerun all time)
    chem_sim_mat = calculate_similarity_matrix(data['Smile'].to_list())

    # create a heatmap
    compounds, resorted_compounds, cluster_accumulation, resorted_mat = clustered_heatmap\
                                                                    (mat = chem_sim_mat,
                                                                     compounds = np.array(data['HETI'].tolist()),
                                                                     title = 'Sample Heatmap',
                                                                     method = 'single',
                                                                     metric = 'euclidean',
                                                                     thresh = 0.9,
                                                                     colored_ticks=data.head(3)['HETI'].to_list())


    # get some substrings
    cluster_2 = cluster_accumulation[1].tolist()

    cluster_2_smiles = data.query('HETI == @cluster_2')['Smile'].to_list()

    common_smile, smart_string = common_substructure(smile_list = cluster_2_smiles,
                                                     thresh = 0.8,
                                                     timeout=60, compare_first=False)

    # quick vis. of smiles: https://pubchem.ncbi.nlm.nih.gov//edit3/index.html