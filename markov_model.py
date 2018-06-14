import argparse
import logging
import os
import pickle
import sys
import warnings

import numpy as np
from numpy import linalg as la
import random
from scipy.stats import rankdata
from scipy.stats import gaussian_kde
from sklearn.neighbors import kneighbors_graph
from hmmlearn import hmm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from bisect import bisect_left
plt.style.use("ggplot")

# Log everything to stdout
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)

def get_parser():
    '''get_parser returns the arg parse object, for use by an external application (and this script)
    '''
    parser = argparse.ArgumentParser(
    description="Generating diffusion maps of trajectories.")

    ################################################################################
    # General Simulation Parameters
    ################################################################################

    parser.add_argument("--seed",
                        dest='seed',
                        help="Seed for Markov model.",
                        type=int,
                        default=1)

    parser.add_argument("--n_clusters",
                        dest='n_clusters',
                        help="n_clusters for Markov model.",
                        type=int,
                        default=15)

    parser.add_argument("--method",
                        dest='method',
                        help="method for Markov model. Can be hmm or agg_clustering.",
                        type=str,
                        default="hmm")

    parser.add_argument("--n_iter",
                        dest='n_iter',
                        help="n_iter for Markov model.",
                        type=int,
                        default=100)

    parser.add_argument("--covariance_type",
                        dest='covariance_type',
                        help="covariance_type for Markov model. Can be spherical, diag, or full.",
                        type=str,
                        default='spherical')

    parser.add_argument("--tol",
                        dest='tol',
                        help="tol for Markov model.",
                        type=float,
                        default=1e-2)

    # Does the user want to quiet output?
    parser.add_argument("--quiet",
                        dest='quiet',
                        action="store_true",
                        help="Turn off logging (debug and info)",
                        default=False)

    parser.add_argument("--input_file_path",
                        dest='input_file_path',
                        type=str,
                        help="input file full path",
                        default=None)

    parser.add_argument("--output_file_path",
                        dest='output_file_path',
                        type=str,
                        help="Output file full path",
                        default=None)
    return parser

################################################################################
#### Supporting Functions
################################################################################

def sorted_eigs(e_vals,e_vecs):
    '''
    Then sort the eigenvectors and eigenvalues
    s.t. the eigenvalues monotonically decrease.
    '''
    l = zip(e_vals,e_vecs.T)
    l = sorted(l,key = lambda z: -z[0])
    return np.asarray([el[0] for el in l]),np.asarray([el[1] for el in l]).T

def make_markov_model(labels,n_clusters):
    T = np.zeros((n_clusters,n_clusters))
    for i in range(len(labels)-1):
        j = i + 1
        T[labels[i],labels[j]] += 1
    row_norm = np.squeeze(np.asarray(T.sum(axis = 1)))
    row_norm = np.power(row_norm,-1)
    row_norm = np.nan_to_num(row_norm)
    return (T.T*row_norm).T

def next_state(state,T_cum):
    r = np.random.uniform()
    return bisect_left(np.asarray(T_cum)[state], r)

def run_markov_chain(start_cluster, T, steps = 10000, slow_down=1):
    '''
    Run Markov chain for a single trajectory.
    '''
    T_cum = np.matrix(np.zeros_like(T)) ## CDF in indices

    for i in range(T.shape[1]):
        s = 0.
        for j in range(T.shape[0]):
            s += T[i,j]
            T_cum[i,j] = s

    row_sums = np.asarray([T[i,:].sum() for i in range(T.shape[1])])
    ## The row sums can sometimes end up not being 1.
    ## Assuming they are >=0, we fix that here.
    for row, row_sum in enumerate(row_sums):
        if row_sum == 0.: ## We can't divide by zero
            T_cum[row] = 1./T_cum.shape[1]
        else:
            T_cum[row] /= row_sum

    current = start_cluster
    outs = []
    for i in range(steps):
        for re in range(slow_down):
            outs.append(current)
        current = next_state(current,T_cum)
    return np.asarray(outs)

def get_cluster_labels(X,n_clusters = 30, num_nearest_neighbors = 100):
    knn_graph = kneighbors_graph(X, num_nearest_neighbors, include_self=False)
    model = AgglomerativeClustering(linkage='ward',
                                    connectivity=knn_graph,
                                    n_clusters=n_clusters)
    model.fit(X)
    labels = model.labels_
    return labels

def get_clusters(labels,n_clusters = 30):
    clusters = [[] for _ in range(n_clusters)]
    for point,cluster in enumerate(labels):
        clusters[cluster].append(point)
    return clusters

def get_hmm_hidden_states(X,
                          n_clusters = 30,
                          return_model = False,
                          n_iter=100,
                          tol=1e-2,
                          covariance_type = 'full',
                          random_state = 1,
                          verbose = False,
                          Ntraj = None):
    if Ntraj is None:
        Ntraj = 1
    assert X.shape[0] % Ntraj == 0
    lengths = [X.shape[0] // Ntraj] * Ntraj
    hmm_model = hmm.GaussianHMM(n_components=n_clusters,
                                covariance_type=covariance_type,
                                n_iter = n_iter,
                                tol = tol,
                                random_state=random_state)
    hmm_model.fit(X,lengths)
    if verbose:
        print("converged", hmm_model.monitor_.converged)
    hidden_states = hmm_model.predict(X)
    if not return_model:
        return hidden_states
    else:
        return hidden_states, hmm_model

def get_obs_sample_points(obs_indices,traj_expects):
    '''
    Observed quantities at sampled points
    '''
    return np.asarray([traj_expects[:,l] for l in obs_indices])

def get_expect_in_clusters(obs_indices,clusters, n_clusters, obs_sample_points):
    '''
    Get the average expectation value in each cluster.
    '''
    expect_in_clusters = {}
    for obs_index in obs_indices:
        expect_in_clusters[obs_index] = [0. for _ in range(n_clusters)]
        for clust_index,cluster in enumerate(clusters):
            for point in cluster:
                expect_in_clusters[obs_index][clust_index] += obs_sample_points[obs_index][point]
            if len(cluster) != 0:
                expect_in_clusters[obs_index][clust_index] /= float(len(cluster))
            else:
                expect_in_clusters[obs_index][clust_index] = None
    return expect_in_clusters

def get_obs_generated(obs_indices,
                      T_matrix, ## Transition matrix used
                      expect_in_clusters,
                      steps = 10000,
                      n_clusters = 10,
                      start_cluster = 0, ## index of starting cluster
                      slow_down=1,
                      return_state_indices_only = False):
    steps = run_markov_chain(start_cluster,T_matrix, steps = steps, slow_down = slow_down)
    if return_state_indices_only:
        return steps
    obs_generated = np.asarray([[expect_in_clusters[l][cluster] for cluster in steps ] for l in obs_indices])
    return obs_generated

def get_reduced_model_time_series(expect_in_clusters,indices,point_in_which_cluster):
    '''
    Return the average expectation value over clusters
    as a time series of the true trajectory
    '''
    return [[expect_in_clusters[l][point_in_which_cluster[point]]
                for point in range(len(indices))] for l in obs_indices]

################################################################################
#### Plotting Functions
################################################################################

def contour_plot(Mat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(abs(Mat), interpolation='nearest')
    fig.colorbar(cax)
    plt.show()

def make_density_scatterplts(X,Y,label_shift = 0, observable_names = None,s=5):
    if observable_names is None:
        observable_names = [str(j+1) for j in range(X.shape[-1])]
    fig, ax = plt.subplots(nrows=X.shape[-1],ncols=Y.shape[-1],figsize = (Y.shape[-1]*10,X.shape[-1]*10))
    for i,row in enumerate(ax):
        for j,col in enumerate(row):
            col.set_title( observable_names[j] +" vs  Phi" + str(i+1+label_shift))
            x = X[:,i]
            y = Y[:,j]

            xy = np.vstack([x,y])
            z = gaussian_kde(xy)(xy)
            col.scatter(x, y, c=np.log(z), s=s, edgecolor='')
    plt.show()

def ellipses_plot(X, indices, hmm_model, n_clusters, std_dev = 1):
    # Calculate the point density
    ### from http://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    x = X[:,indices[0]]
    y = X[:,indices[1]]

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    covariances = np.asarray([[[hmm_model.covars_[clus,n,m]
                                for n in indices]
                                    for m in indices]
                                        for clus in range(n_clusters)])
    means = np.asarray([[hmm_model.means_[clus,n]
                                for n in indices]
                                    for clus in range(n_clusters)])

    sorted_eig_lst = [sorted_eigs(*la.eig(cov)) for cov in covariances]
    angles = np.rad2deg(np.asarray([np.arctan2(*v[1][:,0]) for v in sorted_eig_lst]))
    widths = [2*std_dev*np.sqrt(v[0][0]) for v in sorted_eig_lst]
    heights = [2*std_dev*np.sqrt(v[0][1]) for v in sorted_eig_lst]

    es = [Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor='black', facecolor='none',linewidth = 1)
             for mean,width,height,angle in zip(means,widths,heights,angles) ]

    fig = plt.figure(0,figsize= (10,10))
    ax = fig.add_subplot(111, )
    ax.set_title("coordinates "+str(indices[0]+1)  +","+str(indices[1]+1) )

    for e in es:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_clip_box(ax.bbox)

    # ax.set_xlim(-.06, .06 )
    # ax.set_ylim(-.06,.06 )
    ax.scatter(x, y, c=np.log(z), s=100, edgecolor='')

################################################################################
#### Object used to hold the model
################################################################################

class dim_red:
    '''
    Minimal container for dimensionality reduction.
    '''
    def __init__(self, X, Ntraj, expects_sampled, name, obs_indices=None):
        self.X = X
        self.Ntraj = Ntraj
        self.expects_sampled = np.concatenate(expects_sampled, axis = 0)
        if obs_indices is None:
            self.obs_indices = range(expects_sampled.shape[-1])
        else:
            self.obs_indices = obs_indices
        self.name = name

    def plot_obs_v_diffusion(self,
                            observable_names = None,
                            s=5,
                            which_obs = None):
        if which_obs is None:
            which_obs = range(self.expects_sampled.shape[-1])
        make_density_scatterplts(self.X,
                                self.expects_sampled[:,which_obs],
                                label_shift = 0,
                                observable_names = observable_names,
                                s=s)

    def plot_diffusion_v_diffusion(self,
                                   color='percentile',
                                   max_coord1=4,
                                   max_coord2=4,
                                   observable_names=None):
        if observable_names is None:
            observable_names = [str(i) for i in self.obs_indices]
        for l in self.obs_indices:
            fig = plt.figure(figsize=(max_coord2*10,max_coord1*10))
            if color == 'percentile':
                cols = rankdata(self.expects_sampled[:,l], "average") / self.num_sample_points
            else:
                assert color == 'density'
            for k in range(max_coord1):
                for i in range(k+1,max_coord2):
                    x,y = self.X[:,k],self.X[:,i]
                    ax = fig.add_subplot(max_coord1, max_coord2, k*max_coord2+i+1)
                    if color == 'density':
                        xy = np.vstack([x,y])
                        z = gaussian_kde(xy)(xy)
                        cols = np.log(z)
                        plt.title("Log(density). Phi" + str(i+1) + " versus  Phi" + str(k+1) )
                    else:
                        plt.title("Observable: " + observable_names[l] + "; Coordinates: Phi" + str(i+1) + " versus  Phi" + str(k+1) )
                    plt.scatter(x,y, c = cols)
                    plt.tight_layout()
            plt.show()
            if color == 'density': ## only one iteration
                break

    def plot_diffusion_3d(self,
                          color_by_percentile = True,
                          coords = [0,1,2]):
        if len(coords) != 3:
            raise ValueError("number of coordinates must be 3 for 3D plot.")

        num_obs = len(self.obs_indices)
        fig = plt.figure(figsize=(num_obs*10,10))

        for l in self.obs_indices:
            ax = fig.add_subplot(1, num_obs, l+1, projection='3d')
            expects_sampled_percentile = rankdata(self.expects_sampled[:,l], "average") / self.num_sample_points
            ax.scatter(self.X[:,coords[0]],self.X[:,coords[1]],self.X[:,coords[2]],
                        c = expects_sampled_percentile)
        plt.show()

class markov_model_builder:
    def __init__(self, dim_red=None, name=None):
        if dim_red is not None:
            try:
                self.X = dim_red.X
            except:
                print('Warning: did not find reduced coordinates X in dim_red')
            self.Ntraj = dim_red.Ntraj
            self.expects_sampled = dim_red.expects_sampled
            self.obs_indices = dim_red.obs_indices
            if name is None:
                self.name = dim_red.name + "_markov_builder"
            else:
                self.name = name
            self.status = 'not attempted'
        else:
            print("Warning, dim_red initialized to None.")

    def load(self, file_path):
        f = open(file_path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self, file_path):
        f = open(file_path,'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def build_model(self,
                    n_clusters = 10,
                    method = 'hmm',
                    n_iter=1000,
                    covariance_type = 'full',
                    tol=1e-2,
                    get_expects = True,
                    random_state = 1,
                    verbose = True,
                    which_coords = 'X',
                    coords_indices_to_use = None):
        '''
        method can be 'hmm' or 'agg_clustering'.
        which_coords can be 'X' for reduced coordiantes (e.g. diffusion coords)
        or 'expects' for expectation values
        '''
        if method == 'hmm' or method == 'agg_clustering':
            self.method = method
        else:
            raise ValueError("Unknown method type. method can be 'hmm' or 'agg_clustering'. ")
        self.n_clusters = n_clusters
        self.random_state = random_state

        if which_coords == 'X':
            self.X_to_use = self.X
        else:
            assert which_coords == 'expects'
            self.X_to_use = self.expects_sampled
        if not coords_indices_to_use is None:
            self.X_to_use = self.X_to_use[:,coords_indices_to_use]
        if self.method == 'hmm':
            self.labels, self.hmm_model = get_hmm_hidden_states(self.X_to_use,
                                                                self.n_clusters,
                                                                return_model=True,
                                                                n_iter=n_iter,
                                                                tol=tol,
                                                                covariance_type = covariance_type,
                                                                random_state=random_state,
                                                                verbose = True,
                                                                Ntraj = self.Ntraj)
            self.clusters = get_clusters(self.labels,self.n_clusters)
            self.T = make_markov_model(self.labels,self.n_clusters)
            self.status = 'model built'
        elif self.method == 'agg_clustering':
            self.labels = get_cluster_labels(self.X_to_use, self.n_clusters)
            self.clusters = get_clusters(self.labels,self.n_clusters)
            self.T = make_markov_model(self.labels,self.n_clusters)
            self.status = 'model built'
        else:
            raise ValueError("Unknown method type. method can be 'hmm' or 'agg_clustering'. ")

        if get_expects:
            self.obs_sample_points = get_obs_sample_points(self.obs_indices,self.expects_sampled)
            self.expects_in_clusters = get_expect_in_clusters(self.obs_indices,self.clusters, self.n_clusters, self.obs_sample_points)

    def get_ordering_by_obs(self,obs_index = 0):
        assert self.status == 'model built'
        obs_used = self.expects_sampled[:,obs_index]
        expects_in_clusters = [np.average([obs_used[i] for i in self.clusters[k]]) for k in range(self.n_clusters) ]
        D = {num:i for num,i in  zip(expects_in_clusters,range(self.n_clusters))}
        cluster_order = [D[key] for key in sorted(D.keys())]
        return cluster_order

    def plot_transition_matrix(self, obs_index = None,):
        assert self.status == 'model built'
        if obs_index is None:
            contour_plot(self.T)
        elif isinstance(obs_index,int):
            cluster_order = self.get_ordering_by_obs(obs_index)
            def order_indices(mat):
                return np.asmatrix([[mat[i,j] for i in cluster_order] for j in cluster_order])
            contour_plot(order_indices(self.T))
        else:
            raise ValueError("obs_index should be None or an integer (representing an observable).")

    def ellipses_plot(self,indices = [0,1]):
        '''
        Works only for hmm, using the diffusion coordinates X.
        '''
        assert self.status == 'model built'
        ellipses_plot(self.X_to_use,indices,self.hmm_model,self.n_clusters)

    def generate_obs_traj(self,
                          steps=10000,
                          random_state=1,
                          start_cluster=0,
                          slow_down=1,
                          return_state_indices_only=False,
                          obs_indices=None):
        assert self.status == 'model built'
        if obs_indices is None:
            obs_indices = self.obs_indices
        np.random.seed(random_state)
        return get_obs_generated(obs_indices,
                                self.T, ## Transition matrix used
                                self.expects_in_clusters,
                                steps = steps,
                                n_clusters = self.n_clusters,
                                start_cluster = start_cluster, ## index of starting cluster
                                slow_down=slow_down,
                                return_state_indices_only = return_state_indices_only)

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    output_file_path = args.output_file_path
    input_file_path = args.input_file_path
    seed = args.seed

    n_clusters = args.n_clusters
    method = args.method
    n_iter = args.n_iter
    covariance_type = args.covariance_type
    tol = args.tol

    pkl_file = open(input_file_path, 'rb')
    data1 = pickle.load(pkl_file)

    Ntraj = len(data1['traj_list'])

    if len(data1['expects'].shape) == 2:
        ## data1['expects'].shape = points_total, observables
        ## reshape into Ntraj, points_per_traj, observables
        expects_sampled = data1['expects'].reshape(Ntraj,
                                                 int(data1['expects'].shape[0]/Ntraj),
                                                 data1['expects'].shape[-1])
    elif len(data1['expects'].shape) == 1:
        ## data1['expects'].shape = points_total * observables,
        ## This resulted from a bug in some datasets, which has been fixed.
        ## The trajectories should have been fixed, but include this here just in case.
        num_expects = int((data1['expects']).shape[0] / data1['times'].shape[0])
        expects_sampled = data1['expects'].reshape(Ntraj,
                                                  int(data1['expects'].shape[0]/(Ntraj*num_expects)),
                                                  num_expects)
    else:
        raise ValueError("Unknown data expects shape.")

    vals_tmp, vecs_tmp = data1['vals'][1:], data1['vecs'][:,1:]
    vals, vecs = sorted_eigs(vals_tmp, vecs_tmp)

    name = input_file_path
    dim_red_obj = dim_red(vecs, Ntraj, expects_sampled, name)

    mod = markov_model_builder(dim_red_obj)

    ## Sometimes hmmlearn raises a ValueError for a bad seed:
    ## ValueError: startprob_ must sum to 1.0 (got nan)
    ## Below we handle this. The successful seed value is recorded in
    ## the markov_model_builder class.
    tries = 0
    max_tries = 10
    while tries < max_tries:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=DeprecationWarning)
                mod.build_model(n_clusters=n_clusters,
                                method=method,
                                n_iter=n_iter,
                                covariance_type=covariance_type,
                                tol=tol,
                                get_expects=True,
                                random_state=seed,
                                verbose=True,
                                which_coords='X',
                                coords_indices_to_use=None)
        except ValueError:
            seed += 1
            tries

    mod.save(output_file_path)


    # load_trajectory('diffusion_map_fb1b6ff0eb4b6940e9312f91b2b244d82e1dee2129d0e81e1ee6851b1d9d864a.pkl')
    # markov_model_fb1b6ff0eb4b6940e9312f91b2b244d82e1dee2129d0e81e1ee6851b1d9d864a.pkl
    # markov_model_fe1ff70dd598fc2a7203b18f6bd47c6e7410191c86afd86fd423f1f534321299
    # markov_model_0fcfe35345366834103e42b8981e15a37acd84b938c78f1359332d02a0ba9398
