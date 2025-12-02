import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from collections import Counter

class Recommender:
    def fit(self, train_df):
        raise NotImplementedError
    
    def recommend(self, user_id, n=20, already_seen=None):
        raise NotImplementedError

class PopularityRecommender(Recommender):
    def __init__(self, exclude_top_percent=0.0):
        self.popular_items = []
        self.exclude_top_percent = exclude_top_percent
        
    def fit(self, train_df):
   
        item_counts = train_df['item_id'].value_counts().reset_index()
        item_counts.columns = ['item_id', 'count']
        

        item_counts = item_counts.sort_values('count', ascending=False)
        

        if self.exclude_top_percent > 0:
            num_exclude = int(len(item_counts) * self.exclude_top_percent)
            item_counts = item_counts.iloc[num_exclude:]
            
        self.popular_items = item_counts['item_id'].tolist()
        
    def recommend(self, user_id, n=20, already_seen=None):
        if already_seen is None:
            already_seen = set()
            
        recs = []
        for item in self.popular_items:
            if item not in already_seen:
                recs.append(item)
                if len(recs) == n:
                    break
        return recs

class ItemCFRecommender(Recommender):
    def __init__(self, similarity_metric='cosine', k_neighbors=50):
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.item_sim_matrix = None
        self.user_item_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_to_idx = {}
        
    def fit(self, train_df):
   
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}
        
        rows = train_df['user_id'].map(self.user_to_idx)
        cols = train_df['item_id'].map(self.item_to_idx)
        data = np.ones(len(train_df))
        
        self.user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        
  
        item_user_matrix = self.user_item_matrix.T
        
        if self.similarity_metric == 'cosine':
            if self.k_neighbors is not None and self.k_neighbors > 0:
                from sklearn.neighbors import NearestNeighbors
               
                nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='cosine', algorithm='brute', n_jobs=-1)
                nbrs.fit(item_user_matrix)
                distances, indices = nbrs.kneighbors(item_user_matrix)
                

                n_items = item_user_matrix.shape[0]
                rows = np.repeat(np.arange(n_items), self.k_neighbors + 1)
                cols = indices.flatten()

                data = (1 - distances).flatten()
                
                self.item_sim_matrix = csr_matrix((data, (rows, cols)), shape=(n_items, n_items))
                
   
                try:
                    self.item_sim_matrix.setdiag(0)
                except:

                    pass
                self.item_sim_matrix.eliminate_zeros()
            else:
                self.item_sim_matrix = cosine_similarity(item_user_matrix, dense_output=False)

                try:
                    self.item_sim_matrix.setdiag(0)
                except:
                    pass
            
        elif self.similarity_metric == 'jaccard':

            from sklearn.metrics.pairwise import pairwise_distances

            item_user_bool = item_user_matrix.astype(bool)

            dist_matrix = pairwise_distances(item_user_bool, metric='jaccard', n_jobs=-1)
            self.item_sim_matrix = 1 - dist_matrix
            

            np.fill_diagonal(self.item_sim_matrix, 0)
            self.item_sim_matrix = csr_matrix(self.item_sim_matrix)
        

    def recommend(self, user_id, n=20, already_seen=None):
        if already_seen is None:
            already_seen = set()
            
        if user_id not in self.user_to_idx:
      
            return []
            
        u_idx = self.user_to_idx[user_id]

        user_vector = self.user_item_matrix[u_idx].toarray().flatten()
      
        scores = user_vector.dot(self.item_sim_matrix)
        
      
        k = n + len(already_seen) + 50
        top_indices = np.argsort(scores)[::-1][:k]
        
        recs = []
        for idx in top_indices:
            item = self.idx_to_item[idx]
            if item not in already_seen:
                recs.append(item)
                if len(recs) == n:
                    break
        return recs

    def recommend_batch(self, user_ids, n=20, train_interactions=None):

        u_indices = [self.user_to_idx[u] for u in user_ids if u in self.user_to_idx]
        
        if not u_indices:
            return {}
            

        user_vectors = self.user_item_matrix[u_indices]
        

        scores = user_vectors.dot(self.item_sim_matrix)
        
        results = {}
        
        valid_users = [u for u in user_ids if u in self.user_to_idx]
        
        for i, user_id in enumerate(valid_users):

            user_scores = scores[i].toarray().flatten()
            seen = train_interactions.get(user_id, set()) if train_interactions else set()
            

            k_cand = n + len(seen) + 50
            if k_cand > len(user_scores):
                k_cand = len(user_scores)
                
            top_indices_unsorted = np.argpartition(user_scores, -k_cand)[-k_cand:]
            top_scores = user_scores[top_indices_unsorted]
            

            sorted_indices_local = np.argsort(top_scores)[::-1]
            top_indices = top_indices_unsorted[sorted_indices_local]
            
            recs = []
            for idx in top_indices:
                item = self.idx_to_item[idx]
                if item not in seen:
                    recs.append(item)
                    if len(recs) == n:
                        break
            results[user_id] = recs
            
        return results

class SVDRecommender(Recommender):
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.user_vecs = None
        self.item_vecs = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    def fit(self, train_df):
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}
        
        rows = train_df['user_id'].map(self.user_to_idx)
        cols = train_df['item_id'].map(self.item_to_idx)
        data = np.ones(len(train_df))
        
        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        

        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_vecs = svd.fit_transform(user_item_matrix)
        self.item_vecs = svd.components_.T 
        
    def recommend(self, user_id, n=20, already_seen=None):
        if already_seen is None:
            already_seen = set()
            
        if user_id not in self.user_to_idx:
            return []
            
        u_idx = self.user_to_idx[user_id]
        user_vec = self.user_vecs[u_idx] 
        

        scores = user_vec.dot(self.item_vecs.T)
        

        k = n + len(already_seen) + 50
        top_indices = np.argsort(scores)[::-1][:k]
        
        recs = []
        for idx in top_indices:
            item = self.idx_to_item[idx]
            if item not in already_seen:
                recs.append(item)
                if len(recs) == n:
                    break
        return recs

    def recommend_batch(self, user_ids, n=20, train_interactions=None):
        u_indices = [self.user_to_idx[u] for u in user_ids if u in self.user_to_idx]
        
        if not u_indices:
            return {}
            
        user_vectors = self.user_vecs[u_indices] 
        scores = user_vectors.dot(self.item_vecs.T) 
        
        results = {}
        valid_users = [u for u in user_ids if u in self.user_to_idx]
        
        for i, user_id in enumerate(valid_users):
            user_scores = scores[i]
            seen = train_interactions.get(user_id, set()) if train_interactions else set()
            
            k_cand = n + len(seen) + 50
            if k_cand > len(user_scores):
                k_cand = len(user_scores)
                
            top_indices_unsorted = np.argpartition(user_scores, -k_cand)[-k_cand:]
            top_scores = user_scores[top_indices_unsorted]
            
            sorted_indices_local = np.argsort(top_scores)[::-1]
            top_indices = top_indices_unsorted[sorted_indices_local]
            
            recs = []
            for idx in top_indices:
                item = self.idx_to_item[idx]
                if item not in seen:
                    recs.append(item)
                    if len(recs) == n:
                        break
            results[user_id] = recs
            
        return results

class Item2VecRecommender(Recommender):
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        
    def fit(self, train_df):

        sentences = []
        current_user = None
        current_playlist = []
     
        
        train_df_sorted = train_df 
        items_str = train_df['item_id'].astype(str)
        users = train_df['user_id']
        

        temp_df = pd.DataFrame({'user': users, 'item': items_str})
        sentences = temp_df.groupby('user')['item'].apply(list).tolist()
        
        print(f"Training Word2Vec on {len(sentences)} playlists...")
        self.model = Word2Vec(sentences=sentences, 
                              vector_size=self.vector_size, 
                              window=self.window, 
                              min_count=self.min_count, 
                              workers=4,
                              sg=1, 
                              seed=42)
        
    def recommend(self, user_id, n=20, already_seen=None):
        if already_seen is None:
            already_seen = set()
            
       
        user_history = [str(item) for item in already_seen if str(item) in self.model.wv]
        
        if not user_history:
            return []
     
        recs_tuples = self.model.wv.most_similar(positive=user_history, topn=n + len(already_seen) + 20)
        
        recs = []
        for item_str, score in recs_tuples:
            item = int(item_str)
            if item not in already_seen:
                recs.append(item)
                if len(recs) == n:
                    break
        return recs

class HybridRecommender(Recommender):
    def __init__(self, models_with_weights):
        
        self.models_with_weights = models_with_weights
        
    def fit(self, train_df):
        for model, weight in self.models_with_weights:
            print(f"Fitting sub-model {type(model).__name__}...")
            model.fit(train_df)
            
    def recommend(self, user_id, n=20, already_seen=None):
   
        k_cand = n * 5 
        
        item_scores = {}
        
        for model, weight in self.models_with_weights:
            recs = model.recommend(user_id, n=k_cand, already_seen=already_seen)
            for rank, item in enumerate(recs):
           
                score = weight * (1.0 / (rank + 1))
                item_scores[item] = item_scores.get(item, 0.0) + score
                

        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_recs = [item for item, score in sorted_items[:n]]
        return final_recs

    def recommend_batch(self, user_ids, n=20, train_interactions=None):
        k_cand = n * 5
        
      
        all_model_recs = []
        for model, weight in self.models_with_weights:
            if hasattr(model, 'recommend_batch'):
                print(f"Batch predicting with {type(model).__name__}...")
                recs = model.recommend_batch(user_ids, n=k_cand, train_interactions=train_interactions)
                all_model_recs.append((recs, weight))
            else:
 
                print(f"Sequential predicting with {type(model).__name__}...")
                recs = {}
                for uid in user_ids:
                    seen = train_interactions.get(uid, set()) if train_interactions else set()
                    recs[uid] = model.recommend(uid, n=k_cand, already_seen=seen)
                all_model_recs.append((recs, weight))
                
        results = {}
        for user_id in user_ids:
            item_scores = {}
            for model_recs, weight in all_model_recs:
                recs = model_recs.get(user_id, [])
                for rank, item in enumerate(recs):
                    score = weight * (1.0 / (rank + 1))
                    item_scores[item] = item_scores.get(item, 0.0) + score
            
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            results[user_id] = [item for item, score in sorted_items[:n]]
            
        return results
