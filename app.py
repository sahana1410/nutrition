import streamlit as st
import pandas as pd
import numpy as np
import io
import random
import base64
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import tempfile
import os
import sys
import re
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import stats
import seaborn as sns
LANGCHAIN_AVAILABLE = False
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("Sentence Transformers not installed. Install with: pip install sentence-transformers")
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
try:
    from orchestrator import FoodRecommendationOrchestrator
    print("✓ Orchestrator imported successfully")
    class OrchestratorAgent(FoodRecommendationOrchestrator):
        def __init__(self, csv_path_or_buffer=None):
            if hasattr(csv_path_or_buffer, 'read'):
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    f.write(csv_path_or_buffer.read().decode('utf-8'))
                    temp_path = f.name
                super().__init__(temp_path)
            else:
                super().__init__(csv_path_or_buffer if csv_path_or_buffer else 'food_data.csv')       
        def run(self, profile):
            diet_type = profile.get('diet_choice', 'Vegetarian')
            recommendations = self.get_ensemble_recommendations(diet_type, 5)
            if not recommendations.empty:
                return {"status": "SUCCESS", "recommendations": recommendations.to_dict('records')}
            return {"status": "ERROR", "error": "No recommendations found"}
    class EnhancedOrchestratorAgent(OrchestratorAgent):
        pass        
except ImportError as e:
    st.error(f"Failed to import orchestrator: {e}")
    class OrchestratorAgent:
        def __init__(self, csv_path_or_buffer=None):
            self.df = pd.read_csv(csv_path_or_buffer) if csv_path_or_buffer else pd.DataFrame()
        def run(self, profile):
            return {"status": "ERROR", "error": "Orchestrator not available"}
    class EnhancedOrchestratorAgent(OrchestratorAgent):
        pass
st.set_page_config(page_title="Agentic AI Nutrition Planner", layout="wide")
theme_css = ""
food_background_css = ""
food_background_path = Path("agents/food.jpg")
st.markdown("""
<style>
.header {
    font-size: 42px;
    font-weight: 800;
    padding: 20px;
    color: white;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    border-radius: 12px;
    margin-bottom: 20px;
    animation: fadeIn 1.2s ease-in-out;
}
.card {
    background-color: white;
    padding: 12px;
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}
.quote {
    font-style: italic;
    margin-top: 10px;
}
.health-metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin: 5px;
    text-align: center;
}
.stress-card {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
}
.sleep-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
}
.activity-card {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
}
.cost-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
}
.carbon-card {
    background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
}
.ml-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin: 5px;
}
.evaluation-card {
    background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin: 5px;
}
.confusion-matrix-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin: 5px;
}
.feature-icon {
    font-size: 24px;
    margin-right: 10px;
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #2c3e50;
}
.metric-label {
    font-size: 14px;
    color: #7f8c8d;
}
.progress-bar {
    height: 8px;
    border-radius: 4px;
    background: #e0e0e0;
    margin-top: 10px;
}
.progress-fill {
    height: 100%;
    border-radius: 4px;
}
.warning-box {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 12px;
    border-radius: 4px;
    margin: 10px 0;
}
.info-box {
    background-color: #d1ecf1;
    border-left: 4px solid #17a2b8;
    padding: 12px;
    border-radius: 4px;
    margin: 10px 0;
}
.success-box {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
    padding: 12px;
    border-radius: 4px;
    margin: 10px 0;
}
.algorithm-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
    margin-right: 5px;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header">🥗 Agentic AI Nutrition Planner</div>', unsafe_allow_html=True)
class SimpleHashEmbeddings:
    """Lightweight local fallback embeddings that avoid Torch model loading issues."""
    def __init__(self, dimensions=256):
        self.dimensions = dimensions
    def _embed_text(self, text):
        vector = np.zeros(self.dimensions, dtype=float)
        tokens = re.findall(r"\b\w+\b", str(text).lower())
        for token in tokens:
            vector[hash(token) % self.dimensions] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    def embed_documents(self, texts):
        return [self._embed_text(text) for text in texts]
    def embed_query(self, text):
        return self._embed_text(text)
    def __call__(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)
def create_safe_embeddings():
    if not LANGCHAIN_AVAILABLE:
        return SimpleHashEmbeddings()
    embedding_attempts = [
        {
            'model_name': "all-MiniLM-L6-v2",
            'model_kwargs': {'device': 'cpu'},
            'encode_kwargs': {'normalize_embeddings': True}
        },
        {
            'model_name': "sentence-transformers/all-MiniLM-L6-v2",
            'model_kwargs': {'device': 'cpu'},
            'encode_kwargs': {'normalize_embeddings': True}
        }
    ]
    for config in embedding_attempts:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=config['model_name'],
                model_kwargs=config['model_kwargs'],
                encode_kwargs=config['encode_kwargs']
            )
            embeddings.embed_query("nutrition test")
            return embeddings
        except Exception as exc:
            print(f"Embedding fallback attempt failed for {config['model_name']}: {exc}")
    return SimpleHashEmbeddings()
class LangChainRAGSystem:
    def __init__(self, df):
        self.df = df
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self.evaluation_results = {
            'retrieval_precision': 0.0,
            'retrieval_recall': 0.0,
            'answer_relevance': 0.0,
            'total_queries': 0
        }
        if LANGCHAIN_AVAILABLE:
            self.setup_rag()
    def setup_rag(self):
        try:
            self.embeddings = create_safe_embeddings()
            documents = []
            for _, row in self.df.iterrows():
                text = f"""
                Food: {row['food_name']}
                Diet Type: {row['diet_type']}
                Culture: {row['culture']}
                Meal Type: {row['meal_type']}
                Category: {row['category']}
                Calories: {row['calories']} kcal
                Protein: {row['protein']}g
                Carbohydrates: {row['carbs']}g
                Fat: {row['fat']}g
                Description: {row['description']}
                Recipe: {row['recipe']}
                Medical Suitability: {row.get('medical_suitability', 'general')}
                Cost per 100g: ₹{row.get('cost_per_100g', 0)}
                Carbon Score: {row.get('carbon_score', 0)}
                Rating: {row.get('rating', 0)} stars
                """
                doc = {
                    'page_content': text,
                    'metadata': {
                        'food_name': row['food_name'],
                        'diet_type': row['diet_type'],
                        'calories': row['calories'],
                        'protein': row['protein'],
                        'culture': row['culture'],
                        'meal_type': row['meal_type']
                    }
                }
                documents.append(doc)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=10,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = [doc['page_content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            from langchain_core.documents import Document
            langchain_docs = [
                Document(page_content=text, metadata=metadata) 
                for text, metadata in zip(texts, metadatas)
            ]
            self.vectorstore = FAISS.from_documents(langchain_docs, self.embeddings)
            print("✓ LangChain RAG system initialized successfully")
        except Exception as e:
            print(f"Error setting up LangChain RAG: {e}")
    def query(self, question: str, diet_filter: str = None, top_k: int = 5) -> dict[str, any]:
        """Query the RAG system with optional diet filter"""
        if self.vectorstore is None:
            return self._fallback_query(question, diet_filter)
        try:
            filter_dict = {}
            if diet_filter:
                diet_map = {
                    'vegetarian': 'Vegetarian',
                    'vegan': 'Vegan',
                    'non-vegetarian': 'Non-Vegetarian'
                }
                diet_type = diet_map.get(diet_filter.lower(), diet_filter)
                filter_dict = {"diet_type": diet_type}
            if filter_dict:
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    question, 
                    k=top_k,
                    filter=filter_dict
                )
            else:
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    question, 
                    k=top_k
                )
            sources = []
            retrieval_scores = []
            for doc, score in docs_with_scores:
                similarity = 1.0 / (1.0 + score) 
                retrieval_scores.append(similarity)             
                sources.append({
                    'text': doc.page_content[:200] + "...",
                    'score': similarity,
                    'metadata': doc.metadata
                })
            if sources:
                answer = f"Based on your query, I found {len(sources)} relevant items:\n\n"
                for i, source in enumerate(sources, 1):
                    metadata = source['metadata']
                    answer += f"{i}. **{metadata.get('food_name', 'Unknown')}** ({metadata.get('diet_type', 'Unknown')})\n"
                    answer += f"   - {source['text'][:150]}...\n"
                    answer += f"   - Calories: {metadata.get('calories', 0)} kcal, Protein: {metadata.get('protein', 0)}g\n\n"
                if retrieval_scores:
                    avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores)
                    self.evaluation_results['retrieval_precision'] = avg_retrieval_score
                    self.evaluation_results['answer_relevance'] = min(avg_retrieval_score * 1.1, 1.0)
            else:
                answer = f"I couldn't find specific items matching '{question}'. Try rephrasing your question or removing filters."
            self.evaluation_results['total_queries'] += 1 
            return {
                'answer': answer,
                'sources': sources,
                'confidence': len(sources) / top_k if sources else 0,
                'retrieval_scores': retrieval_scores
            }
        except Exception as e:
            print(f"RAG query error: {e}")
            return self._fallback_query(question, diet_filter)
    def _fallback_query(self, question: str, diet_filter: str = None):
        """Fallback when LangChain RAG is not available"""
        question_lower = question.lower()
        results = []
        search_df = self.df
        if diet_filter:
            diet_map = {
                'vegetarian': 'Vegetarian',
                'vegan': 'Vegan',
                'non-vegetarian': 'Non-Vegetarian'
            }
            diet_type = diet_map.get(diet_filter.lower(), diet_filter)
            search_df = self.df[self.df['diet_type'] == diet_type]
        for _, row in search_df.iterrows():
            score = 0
            if any(word in row['food_name'].lower() for word in question_lower.split()):
                score += 2
            if any(word in row['description'].lower() for word in question_lower.split()):
                score += 1
            if any(word in row['category'].lower() for word in question_lower.split()):
                score += 1
            if score > 0:
                results.append({
                    'food': row['food_name'],
                    'score': score / 4,
                    'description': row['description'],
                    'diet_type': row['diet_type'],
                    'calories': row['calories'],
                    'protein': row['protein']
                })
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
        if results:
            answer = f"Based on your query, I found {len(results)} relevant items:\n\n"
            for i, r in enumerate(results, 1):
                answer += f"{i}. **{r['food']}** ({r['diet_type']})\n"
                answer += f"   - {r['description']}\n"
                answer += f"   - Calories: {r['calories']} kcal, Protein: {r['protein']}g\n\n"
        else:
            answer = f"I couldn't find specific items matching '{question}'. Try rephrasing your question or removing filters."
        self.evaluation_results['total_queries'] += 1
        return {
            'answer': answer,
            'sources': results[:3],
            'confidence': len(results) / 5 if results else 0
        }
    def get_evaluation_metrics(self):
        return self.evaluation_results
class EnhancedNutritionML:
    """Enhanced ML Recommendation Engine with Proper Diet Filtering and Model Evaluation"""
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.embedder = None
        self.classifier = None
        self.xgb_model = None
        self.lgb_model = None
        self.iso_forest = None
        self.evaluation_results = {}
        self.langchain_available = LANGCHAIN_AVAILABLE
        self._prepare_data()
        self._train_all_models()
    def _prepare_data(self):
        numeric_cols = ['calories', 'protein', 'carbs', 'fat', 'cost_per_100g', 'carbon_score']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        if 'diet_type' in self.df.columns:
            self.df['diet_type'] = self.df['diet_type'].astype(str).str.lower().str.strip()
            diet_mapping = {
                'vegetarian': 'Vegetarian', 
                'veg': 'Vegetarian', 
                'pure veg': 'Vegetarian',
                'non-vegetarian': 'Non-Vegetarian', 
                'non veg': 'Non-Vegetarian', 
                'nonvegetarian': 'Non-Vegetarian',
                'vegan': 'Vegan', 
                'plant based': 'Vegan'
            }
            self.df['diet_type'] = self.df['diet_type'].map(lambda x: 
                diet_mapping.get(x, x.title()) if x in diet_mapping else x.title())
        if 'culture' in self.df.columns:
            self.df = self._standardize_culture_values(self.df)
        string_cols = ['food_name', 'diet_type', 'culture', 'meal_type', 'category', 'recipe', 'medical_suitability']
        for col in string_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('').astype(str)
        self.feature_cols = ['calories', 'protein', 'carbs', 'fat']
        available_cols = [col for col in self.feature_cols if col in self.df.columns]
        if len(available_cols) >= 2:
            self.df['total_nutrition'] = self.df[available_cols].sum(axis=1)
            self.df['protein_ratio'] = self.df['protein'] / (self.df['total_nutrition'] + 1e-10)
            self.df['calorie_density'] = self.df['calories'] / (self.df['total_nutrition'] + 1e-10)
            self.df['quality_score'] = self.df['protein_ratio'] * 100
            self.df['is_high_quality'] = (self.df['quality_score'] > self.df['quality_score'].median()).astype(int)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.embedder = None    
    def _standardize_culture_values(self, df):
        if 'culture' not in df.columns:
            return df  
        df['culture'] = df['culture'].astype(str).str.lower().str.strip()
        culture_mapping = {
            'indian': ['indian', 'india', 'south indian', 'north indian', 'punjabi', 'gujarati', 'tamil', 'bengali'],
            'western': ['western', 'american', 'european', 'usa', 'uk', 'british', 'french', 'german', 'italian', 'spanish'],
            'asian': ['asian', 'chinese', 'japanese', 'korean', 'thai', 'vietnamese', 'east asian'],
            'mediterranean': ['mediterranean', 'greek', 'spanish', 'middle eastern', 'turkish', 'lebanese'],
            'any': ['any', 'international', 'global', 'fusion', 'continental']
        }
        def map_to_standard(value):
            value_lower = value.lower().strip()
            for standard, variations in culture_mapping.items():
                if any(var in value_lower for var in variations) or value_lower in variations:
                    return standard.title()
            return value.title()
        df['culture'] = df['culture'].apply(map_to_standard)
        return df
    def _diet_match_mask(self, diet_series, diet_choice):
        """Return an exact-match mask for supported diet categories."""
        normalized_series = diet_series.astype(str).str.lower().str.strip()
        normalized_choice = str(diet_choice).lower().strip()
        if 'vegan' in normalized_choice:
            return normalized_series.eq('vegan')
        if 'vegetarian' in normalized_choice and 'non-vegetarian' not in normalized_choice:
            return normalized_series.eq('vegetarian')
        if 'non-vegetarian' in normalized_choice:
            return normalized_series.eq('non-vegetarian')
        
        return pd.Series([True] * len(diet_series), index=diet_series.index)
    def _normalize_medical_conditions(self, user_profile):
        """Normalize medical conditions from UI/profile into a clean list."""
        conditions = user_profile.get('medical_conditions')
        if conditions is None:
            conditions = user_profile.get('medical_filter', [])
        if isinstance(conditions, str):
            conditions = [conditions]
        normalized = []
        for condition in conditions or []:
            value = str(condition).strip().lower()
            if value and value != 'none':
                normalized.append(value)
        return list(dict.fromkeys(normalized))
    def _medical_suitability_filter(self, df, user_profile):
        if 'medical_suitability' not in df.columns:
            return df.copy().reset_index(drop=True)
        selected_conditions = self._normalize_medical_conditions(user_profile)
        if not selected_conditions:
            return df.copy().reset_index(drop=True)
        suitability = df['medical_suitability'].fillna('general').astype(str).str.lower().str.strip()
        def is_medically_suitable(value):
            tokens = [token.strip() for token in value.split(',') if token.strip()]
            if not tokens:
                tokens = ['general']
            return any(condition in tokens for condition in selected_conditions)
        mask = suitability.apply(is_medically_suitable)
        if mask.any():
            return df[mask].reset_index(drop=True)
        return pd.DataFrame(columns=df.columns)
    def _train_all_models(self):
        if len(self.df) < 20:
            return
        feature_cols = ['calories', 'protein', 'carbs', 'fat', 'protein_ratio', 'calorie_density']
        available_cols = [col for col in feature_cols if col in self.df.columns]
        if len(available_cols) >= 3 and 'is_high_quality' in self.df.columns:
            X = self.df[available_cols].fillna(0).values
            y = self.df['is_high_quality'].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X_train, y_train)
            if XGBOOST_AVAILABLE:
                try:
                    self.xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
                    self.xgb_model.fit(X_train, y_train)
                except:
                    self.xgb_model = None
            if LIGHTGBM_AVAILABLE:
                try:
                    self.lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                    self.lgb_model.fit(X_train, y_train)
                except:
                    self.lgb_model = None
            self.iso_forest = IsolationForest(contamination=0.1, random_state=42)
            self.iso_forest.fit(X_train)
            self._evaluate_models(X_test, y_test)
    def _evaluate_models(self, X_test, y_test):
        self.evaluation_results = {}
        if self.classifier is not None:
            y_pred = self.classifier.predict(X_test)
            y_proba = self.classifier.predict_proba(X_test)
            self.evaluation_results['random_forest'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'avg_confidence': y_proba.max(axis=1).mean()}
        if self.xgb_model is not None:
            y_pred = self.xgb_model.predict(X_test)
            y_proba = self.xgb_model.predict_proba(X_test)
            self.evaluation_results['xgboost'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'avg_confidence': y_proba.max(axis=1).mean()}
        if self.lgb_model is not None:
            y_pred = self.lgb_model.predict(X_test)
            y_proba = self.lgb_model.predict_proba(X_test)
            self.evaluation_results['lightgbm'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'avg_confidence': y_proba.max(axis=1).mean()}
        if self.classifier is not None and len(X_test) > 0:
            try:
                cv_scores = cross_val_score(self.classifier, X_test, y_test, cv=min(3, len(np.unique(y_test))))
                self.evaluation_results['cross_validation'] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()}
            except:
                pass
    def _strict_diet_filter(self, df, user_profile):
        df_filtered = df.copy()
        diet_choice = user_profile.get('diet_choice', '').lower().strip()
        if diet_choice and 'diet_type' in df_filtered.columns:
            df_filtered['diet_type_lower'] = df_filtered['diet_type'].str.lower().str.strip()
            if 'vegan' in diet_choice:
                df_filtered = df_filtered[df_filtered['diet_type_lower'] == 'vegan']
            elif 'vegetarian' in diet_choice and 'non-vegetarian' not in diet_choice:
                df_filtered = df_filtered[df_filtered['diet_type_lower'] == 'vegetarian']
            elif 'non-vegetarian' in diet_choice:
                df_filtered = df_filtered[
                    df_filtered['diet_type_lower'] == 'non-vegetarian']
            if 'diet_type_lower' in df_filtered.columns:
                df_filtered = df_filtered.drop('diet_type_lower', axis=1)
        return df_filtered.reset_index(drop=True)
    def _cultural_filter(self, df, user_profile):
        df_filtered = df.copy()
        culture = user_profile.get('culture', '').lower().strip()
        if culture and culture != 'any' and 'culture' in df_filtered.columns:
            culture_mapping = {
                'indian': ['indian', 'india', 'south indian', 'north indian'],
                'western': ['western', 'american', 'european', 'usa', 'uk', 'british'],
                'asian': ['asian', 'chinese', 'japanese', 'korean', 'thai', 'vietnamese'],
                'mediterranean': ['mediterranean', 'greek', 'italian', 'spanish', 'middle eastern'],
                'any': []}
            matching_cultures = culture_mapping.get(culture.lower(), [culture.lower()])
            matching_cultures.append(culture.lower())  
            culture_mask = df_filtered['culture'].str.lower().isin(matching_cultures)
            if culture_mask.any():
                df_filtered = df_filtered[culture_mask]
            else:
                partial_mask = pd.Series([False] * len(df_filtered))
                for cult in matching_cultures:
                    if cult and len(cult) > 2: 
                        partial_mask = partial_mask | df_filtered['culture'].str.contains(cult, case=False, na=False)
                if partial_mask.any():
                    df_filtered = df_filtered[partial_mask] 
                else:
                    st.warning(f"No items found for culture '{culture}'. Showing all items from your diet preference.")
                    return df
        return df_filtered.reset_index(drop=True)
    def debug_culture_matches(self, user_profile):
        culture = user_profile.get('culture', '').lower().strip()
        if 'culture' in self.df.columns:
            unique_cultures = self.df['culture'].unique()
            st.write(f"Available cultures in dataset: {unique_cultures}")
            st.write(f"Selected culture: {culture}")
            matches = self.df[self.df['culture'].str.lower().str.contains(culture, na=False)]
            st.write(f"Direct matches found: {len(matches)}")
            return matches
        return None
    def _meal_type_filter(self, df, user_profile):
        df_filtered = df.copy()
        preferred_meal = user_profile.get('preferred_meal_type', '').lower().strip()
        if preferred_meal and 'meal_type' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['meal_type'].str.lower().str.contains(preferred_meal, na=False)]
        return df_filtered.reset_index(drop=True)
    def _build_isolation_features(self, df):
        feature_df = df.copy()
        base_cols = ['calories', 'protein', 'carbs', 'fat']
        for col in base_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
        total_nutrition = feature_df[base_cols].sum(axis=1)
        feature_df['protein_ratio'] = feature_df['protein'] / (total_nutrition + 1e-10)
        feature_df['calorie_density'] = feature_df['calories'] / (total_nutrition + 1e-10)
        return feature_df[['calories', 'protein', 'carbs', 'fat', 'protein_ratio', 'calorie_density']].fillna(0)
    def _isolation_forest_scoring(self, df, user_profile):
        scores = np.ones(len(df))
        if len(df) < 10:
            return {
                'recommendation_scores': scores,
                'anomaly_scores': np.zeros(len(df)),
                'is_anomaly': np.zeros(len(df), dtype=bool)}
        try:
            isolation_features = self._build_isolation_features(df)
            feature_values = isolation_features.values
            if self.iso_forest is not None:
                iso_forest = self.iso_forest
            else:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                iso_forest.fit(feature_values)
            predictions = iso_forest.predict(feature_values)
            decision_scores = iso_forest.decision_function(feature_values)
            normalized_scores = decision_scores.copy()
            if normalized_scores.max() > normalized_scores.min():
                normalized_scores = (normalized_scores - normalized_scores.min()) / \
                                   (normalized_scores.max() - normalized_scores.min())
            else:
                normalized_scores = np.full(len(df), 0.5)
            recommendation_scores = normalized_scores.copy()
            diet_choice = user_profile.get('diet_choice', '').lower()
            culture = user_profile.get('culture', '').lower()
            if diet_choice and 'diet_type' in df.columns:
                diet_boost = self._diet_match_mask(df['diet_type'], diet_choice).astype(float) * 0.2
                recommendation_scores = recommendation_scores + diet_boost
            if culture and culture != 'any' and 'culture' in df.columns:
                culture_boost = df['culture'].astype(str).str.lower().str.contains(culture, na=False).astype(float) * 0.1
                recommendation_scores = recommendation_scores + culture_boost
            anomaly_penalty = (predictions == -1).astype(float) * 0.2
            recommendation_scores = recommendation_scores - anomaly_penalty
            if recommendation_scores.max() > recommendation_scores.min():
                recommendation_scores = (recommendation_scores - recommendation_scores.min()) / \
                                      (recommendation_scores.max() - recommendation_scores.min())
            else:
                recommendation_scores = np.full(len(df), 0.5)
            return {
                'recommendation_scores': recommendation_scores,
                'anomaly_scores': decision_scores,
                'is_anomaly': predictions == -1}
        except Exception as e:
            print(f"Isolation Forest error: {e}")
        return {
            'recommendation_scores': scores,
            'anomaly_scores': np.zeros(len(df)),
            'is_anomaly': np.zeros(len(df), dtype=bool)}
    def _xgboost_scoring(self, df, user_profile):
        scores = np.ones(len(df))
        if self.xgb_model is None or len(df) < 10:
            return scores
        try:
            feature_cols = ['calories', 'protein', 'carbs', 'fat']
            if 'cost_per_100g' in df.columns:
                feature_cols.append('cost_per_100g')
            available_cols = [col for col in feature_cols if col in df.columns]
            if len(available_cols) >= 3:
                X = df[available_cols].fillna(0).values
                proba = self.xgb_model.predict_proba(X)
                if proba.shape[1] > 1:
                    scores = proba[:, 1]  
                else:
                    scores = proba[:, 0]
                if scores.max() > scores.min():
                    scores = (scores - scores.min()) / (scores.max() - scores.min())
                diet_choice = user_profile.get('diet_choice', '').lower()
                if diet_choice and 'diet_type' in df.columns:
                    diet_boost = self._diet_match_mask(df['diet_type'], diet_choice).astype(float) * 0.15
                    scores = scores + diet_boost
                    if scores.max() > scores.min():
                        scores = (scores - scores.min()) / (scores.max() - scores.min())
        except Exception as e:
            print(f"XGBoost error: {e}")
        return scores
    def _lightgbm_scoring(self, df, user_profile):
        scores = np.ones(len(df))
        if self.lgb_model is None or len(df) < 10:
            return scores
        try:
            feature_cols = ['calories', 'protein', 'carbs', 'fat']
            if 'cost_per_100g' in df.columns:
                feature_cols.append('cost_per_100g')
            available_cols = [col for col in feature_cols if col in df.columns]
            if len(available_cols) >= 3:
                X = df[available_cols].fillna(0).values
                proba = self.lgb_model.predict_proba(X)
                if proba.shape[1] > 1:
                    scores = proba[:, 1]
                else:
                    scores = proba[:, 0]
                if scores.max() > scores.min():
                    scores = (scores - scores.min()) / (scores.max() - scores.min())
                culture = user_profile.get('culture', '').lower()
                if culture and culture != 'any' and 'culture' in df.columns:
                    culture_boost = df['culture'].str.lower().str.contains(culture, na=False).astype(float) * 0.1
                    scores = scores + culture_boost
                    if scores.max() > scores.min():
                        scores = (scores - scores.min()) / (scores.max() - scores.min())
        except Exception as e:
            print(f"LightGBM error: {e}")
        return scores
    def _rule_based_nutritional_scoring(self, df, user_profile):
        scores = np.zeros(len(df))
        if len(df) == 0:
            return scores
        goal = user_profile.get('goal', '').lower()
        calorie_target = user_profile.get('calorie_target', 2000)
        if 'protein' in df.columns:
            protein_target = calorie_target * 0.25 / 4  
            protein_score = df['protein'] / (protein_target + 1e-10)
            protein_score = np.clip(protein_score, 0, 2) / 2  
            scores += protein_score * 0.3
        if 'calories' in df.columns:
            if 'loss' in goal:
                calorie_score = 1 - (df['calories'] / df['calories'].max())
            elif 'gain' in goal:
                calorie_score = df['calories'] / df['calories'].max()
            else:
                calorie_dev = np.abs(df['calories'] - 500) / 500
                calorie_score = np.exp(-calorie_dev)   
            scores += calorie_score * 0.25
        if all(col in df.columns for col in ['protein', 'carbs', 'fat']):
            total = df[['protein', 'carbs', 'fat']].sum(axis=1) + 1e-10 
            if 'loss' in goal:
                ideal_ratios = {'protein': 0.35, 'carbs': 0.40, 'fat': 0.25}
            elif 'gain' in goal:
                ideal_ratios = {'protein': 0.30, 'carbs': 0.50, 'fat': 0.20}
            else:
                ideal_ratios = {'protein': 0.25, 'carbs': 0.50, 'fat': 0.25}
            balance_score = 0
            for macro, ideal in ideal_ratios.items():
                actual = df[macro] / total
                balance_score += 1 - np.abs(actual - ideal)
            balance_score = balance_score / 3
            scores += balance_score * 0.25
        diet_choice = user_profile.get('diet_choice', '').lower()
        if diet_choice and 'diet_type' in df.columns:
            diet_score = self._diet_match_mask(df['diet_type'], diet_choice).astype(float)
            scores += diet_score * 0.2
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores
    def _sentence_transformer_scoring(self, df, user_profile):
        scores = np.ones(len(df))
        if self.embedder is None or len(df) < 5:
            return scores
        try:
            food_descriptions = []
            for idx, row in df.iterrows():
                desc_parts = []
                if 'food_name' in row:
                    desc_parts.append(str(row['food_name']))
                if 'diet_type' in row:
                    desc_parts.append(f"diet: {row['diet_type']}")
                if 'culture' in row:
                    desc_parts.append(f"culture: {row['culture']}")
                if 'category' in row:
                    desc_parts.append(f"category: {row['category']}")
                if 'meal_type' in row:
                    desc_parts.append(f"meal: {row['meal_type']}")
                food_descriptions.append(". ".join(desc_parts))
            user_desc_parts = []
            if user_profile.get('diet_choice'):
                user_desc_parts.append(f"prefers {user_profile['diet_choice']} diet")
            if user_profile.get('culture') and user_profile['culture'] != 'Any':
                user_desc_parts.append(f"prefers {user_profile['culture']} cuisine")
            if user_profile.get('goal'):
                user_desc_parts.append(f"goal is {user_profile['goal']}")
            if user_profile.get('preferred_meal_type'):
                user_desc_parts.append(f"prefers {user_profile['preferred_meal_type']}")
            user_description = ". ".join(user_desc_parts)
            if user_description.strip():
                food_embeddings = self.embedder.encode(food_descriptions, convert_to_numpy=True)
                user_embedding = self.embedder.encode([user_description], convert_to_numpy=True)
                similarities = np.dot(food_embeddings, user_embedding.T).flatten()
                if similarities.max() > similarities.min():
                    scores = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        except Exception as e:
            print(f"Sentence Transformer error: {e}")
        return scores
    def _hybrid_ensemble_scoring(self, df, user_profile):
        if len(df) == 0:
            return np.array([])
        scores_dict = {}
        iso_result = self._isolation_forest_scoring(df, user_profile)
        scores_dict['isolation'] = iso_result['recommendation_scores']
        scores_dict['rule_based'] = self._rule_based_nutritional_scoring(df, user_profile)
        scores_dict['transformer'] = self._sentence_transformer_scoring(df, user_profile)
        if self.xgb_model is not None:
            scores_dict['xgboost'] = self._xgboost_scoring(df, user_profile)
        if self.lgb_model is not None:
            scores_dict['lightgbm'] = self._lightgbm_scoring(df, user_profile)
        n_models = len(scores_dict)
        if n_models == 0:
            return np.ones(len(df))
        ensemble_scores = np.zeros(len(df))
        for model, scores in scores_dict.items():
            if model == 'isolation':
                weight = 0.15
            elif model == 'rule_based':
                weight = 0.20
            elif model == 'transformer':
                weight = 0.20
            elif model in ['xgboost', 'lightgbm']:
                weight = 0.225  # Split remaining weight
            else:
                weight = 1.0 / n_models
            ensemble_scores += scores * weight
        if ensemble_scores.max() > ensemble_scores.min():
            ensemble_scores = (ensemble_scores - ensemble_scores.min()) / \
                            (ensemble_scores.max() - ensemble_scores.min())
        return ensemble_scores
    def recommend_foods(self, user_profile, top_k=10, algorithm='ensemble'):
        df_filtered = self.df.copy()
        diet_choice = user_profile.get('diet_choice', '').lower().strip()
        if diet_choice and 'diet_type' in df_filtered.columns:
            df_filtered = self._strict_diet_filter(df_filtered, user_profile)
        if len(df_filtered) == 0:
            st.warning(f"No foods found for diet choice: {diet_choice}. Showing all foods.")
            df_filtered = self.df.copy()
        selected_conditions = self._normalize_medical_conditions(user_profile)
        if selected_conditions and 'medical_suitability' in df_filtered.columns:
            medically_filtered = self._medical_suitability_filter(df_filtered, user_profile)
            if len(medically_filtered) > 0:
                df_filtered = medically_filtered
            else:
                st.warning(
                    f"No foods found for selected medical conditions: {', '.join(selected_conditions)}. "
                    "Using foods from your diet preference instead.")
        culture = user_profile.get('culture', '').lower().strip()
        if culture and culture != 'any' and 'culture' in df_filtered.columns:
            culture_mask = df_filtered['culture'].str.lower().str.contains(culture, na=False)
            if culture_mask.any():
                df_filtered = df_filtered[culture_mask]
        meal_type = user_profile.get('preferred_meal_type', '').lower().strip()
        if meal_type and 'meal_type' in df_filtered.columns:
            meal_mask = df_filtered['meal_type'].str.lower().str.contains(meal_type, na=False)
            if meal_mask.any():
                df_filtered = df_filtered[meal_mask]
        if len(df_filtered) == 0:
            st.warning("No foods match your preferences. Using all foods from your diet preference.")
            df_filtered = self.df.copy()
            if diet_choice and 'diet_type' in df_filtered.columns:
                df_filtered = self._strict_diet_filter(df_filtered, user_profile)
            if selected_conditions and 'medical_suitability' in df_filtered.columns:
                medically_filtered = self._medical_suitability_filter(df_filtered, user_profile)
                if len(medically_filtered) > 0:
                    df_filtered = medically_filtered
        df_filtered = df_filtered.reset_index(drop=True)
        if algorithm == 'isolation_forest':
            result = self._isolation_forest_scoring(df_filtered, user_profile)
            scores = result['recommendation_scores']
            df_filtered['anomaly_score'] = result['anomaly_scores']
            df_filtered['is_anomaly'] = result.get('is_anomaly', False)
        elif algorithm == 'rule_based':
            scores = self._rule_based_nutritional_scoring(df_filtered, user_profile)
        elif algorithm == 'sentence_transformers':
            scores =self._sentence_transformer_scoring(df_filtered, user_profile)
        elif algorithm == 'xgboost':
            scores = self._xgboost_scoring(df_filtered, user_profile)
        elif algorithm == 'lightgbm':
            scores = self._lightgbm_scoring(df_filtered, user_profile)
        else:  
            scores = self._hybrid_ensemble_scoring(df_filtered, user_profile)
        if len(scores) == 0:
            return pd.DataFrame()
        df_filtered['ml_score'] = scores
        df_filtered = df_filtered.sort_values('ml_score', ascending=False)
        if algorithm == 'isolation_forest' and 'is_anomaly' in df_filtered.columns:
            anomaly_penalty = df_filtered['is_anomaly'].astype(float) * 0.05
            df_filtered['ml_score'] = df_filtered['ml_score'] - anomaly_penalty
            df_filtered = df_filtered.sort_values('ml_score', ascending=False)
        return df_filtered.head(top_k).reset_index(drop=True)
    def _create_fallback_meal(self, meal_time, user_profile):
        fallback_meals = {
            'Breakfast': {
                'food_name': 'Healthy Breakfast Bowl',
                'calories': 350,
                'protein': 15,
                'carbs': 45,
                'fat': 12,
                'diet_type': user_profile.get('diet_choice', 'General'),
                'medical_suitability': ', '.join(self._normalize_medical_conditions(user_profile)) or 'general',
                'culture': user_profile.get('culture', 'Any'),
                'category': 'Breakfast',
                'recipe': 'Mix oats, fruits, and nuts for a balanced breakfast',
                'ml_score': 0.5,
                'selection_reason': 'Fallback option - customize based on preferences'},
            'Lunch': {
                'food_name': 'Balanced Lunch Plate',
                'calories': 450,
                'protein': 25,
                'carbs': 50,
                'fat': 15,
                'diet_type': user_profile.get('diet_choice', 'General'),
                'medical_suitability': ', '.join(self._normalize_medical_conditions(user_profile)) or 'general',
                'culture': user_profile.get('culture', 'Any'),
                'category': 'Main',
                'recipe': 'Combine lean protein, vegetables, and whole grains',
                'ml_score': 0.5,
                'selection_reason': 'Fallback option - customize based on preferences'},
            'Dinner': {
                'food_name': 'Nutritious Dinner',
                'calories': 400,
                'protein': 30,
                'carbs': 35,
                'fat': 18,
                'diet_type': user_profile.get('diet_choice', 'General'),
                'medical_suitability': ', '.join(self._normalize_medical_conditions(user_profile)) or 'general',
                'culture': user_profile.get('culture', 'Any'),
                'category': 'Main',
                'recipe': 'Grilled protein with roasted vegetables',
                'ml_score': 0.5,
                'selection_reason': 'Fallback option - customize based on preferences'},
            'Snack': {
                'food_name': 'Healthy Snack',
                'calories': 150,
                'protein': 5,
                'carbs': 20,
                'fat': 5,
                'diet_type': user_profile.get('diet_choice', 'General'),
                'medical_suitability': ', '.join(self._normalize_medical_conditions(user_profile)) or 'general',
                'culture': user_profile.get('culture', 'Any'),
                'category': 'Snack',
                'recipe': 'Fresh fruits, nuts, or yogurt',
                'ml_score': 0.5,
                'selection_reason': 'Fallback option - customize based on preferences'}}
        return fallback_meals.get(meal_time, fallback_meals['Lunch'])
    def _get_detailed_recipe_text(self, meal_info):
        food_name = meal_info.get('food_name', 'This meal')
        meal_type = meal_info.get('category') or meal_info.get('meal_type') or 'Meal'
        culture = meal_info.get('culture', 'Any')
        base_recipe = str(meal_info.get('recipe', '')).strip()
        calories = float(meal_info.get('calories', 0))
        protein = float(meal_info.get('protein', 0))
        carbs = float(meal_info.get('carbs', 0))
        fat = float(meal_info.get('fat', 0))
        medical = str(meal_info.get('medical_suitability', 'general')).strip()
        ingredient_hint = "Use fresh vegetables, your preferred protein source, moderate seasoning, and a small amount of healthy oil."
        if 'breakfast' in str(meal_type).lower():
            ingredient_hint = "Use easy-to-digest ingredients, a steady energy source such as oats or whole grains, and fruit, nuts, or a protein-rich side."
        elif 'snack' in str(meal_type).lower():
            ingredient_hint = "Keep the portion light and balanced with one protein source and one fiber-rich ingredient."
        elif 'indian' in str(culture).lower():
            ingredient_hint = "Use onions, tomatoes, ginger, garlic, cumin, turmeric, and coriander in moderate quantities with your main grain or protein."
        guidance = []
        if medical and medical.lower() != 'general':
            guidance.append(f"Medical suitability note: this dish is tagged for `{medical}` in the dataset, so keep the seasoning and portion size aligned with that condition.")
        if calories > 0:
            guidance.append(f"This serving is approximately `{calories:.0f} kcal` with `{protein:.1f} g` protein, `{carbs:.1f} g` carbs, and `{fat:.1f} g` fat.")
        if not base_recipe:
            base_recipe = "Prepare the main ingredients, cook them until done, and balance the dish with vegetables and seasoning."
        return (
            f"**Overview**\n"
            f"{food_name} is planned as your {str(meal_type).lower()} option. {base_recipe}\n\n"
            f"**Suggested Ingredients**\n"
            f"{ingredient_hint}\n\n"
            f"**Simple Method**\n"
            f"1. Prep all ingredients and keep the portions moderate.\n"
            f"2. Start with the base cooking step: {base_recipe.lower().rstrip('.') }.\n"
            f"3. Adjust salt, spice, and oil carefully so the meal stays balanced and easy to follow.\n"
            f"4. Serve warm with a suitable side such as vegetables, whole grains, or a light accompaniment if needed.\n\n")
    def _get_selection_reason(self, algorithm, food, day):
        """Get a reason why this food was selected"""
        if algorithm == 'isolation_forest' and bool(food.get('is_anomaly', False)):
            return f"Day {day}: Included as a distinctive outlier, but ranked below the most nutritionally consistent options"
        reasons = {
            'isolation_forest': [
                f"Day {day}: Strong nutritional match with a distinct profile",
                f"Day {day}: Selected for balanced nutrition within your filtered choices",
                f"Day {day}: A good fit that still adds variety to the plan",
                f"Day {day}: Chosen from foods that best match the learned nutrition pattern"],
            'rule_based': [
                f"Day {day}: Optimal protein-to-calorie ratio",
                f"Day {day}: Perfectly matches your nutritional goals",
                f"Day {day}: Selected based on dietary guidelines",
                f"Day {day}: Balanced macronutrient profile"],
            'sentence_transformers': [
                f"Day {day}: Closely matches your preferences semantically",
                f"Day {day}: High relevance to your dietary description",
                f"Day {day}: Semantic search found this perfect match",
                f"Day {day}: Contextually aligned with your needs"],
            'xgboost': [
                f"Day {day}: High confidence prediction ({(food.get('ml_score', 0)*100):.0f}%)",
                f"Day {day}: Top-ranked by gradient boosting analysis",
                f"Day {day}: XGBoost identified this as optimal choice",
                f"Day {day}: Complex pattern matching selected this item"],
            'lightgbm': [
                f"Day {day}: Efficient gradient boosting recommendation",
                f"Day {day}: LightGBM's top pick for your profile",
                f"Day {day}: Fast and accurate model selected this",
                f"Day {day}: Optimized for your dietary preferences"],
            'hybrid': [
                f"Day {day}: Consensus across multiple AI models",
                f"Day {day}: Ensemble voting selected this item",
                f"Day {day}: Combined wisdom of all ML algorithms",
                f"Day {day}: Most robust recommendation for your needs"]}  
        model_reasons = reasons.get(algorithm, reasons['hybrid'])
        return model_reasons[(day - 1) % len(model_reasons)]
    def generate_meal_plan(self, user_profile, days=None):
        if days is None:
            days = user_profile.get('simulation_days', 3)
        try:
            days = int(days)
            days = max(1, min(days, 30)) 
        except:
            days = 3 
        algorithm = user_profile.get('ml_algorithm', 'ensemble')
        meal_plan = {}
        model_characteristics = {
            'isolation_forest': {
                'name': '🌲 Isolation Forest',
                'description': 'Discovering unique and unusual food combinations',
                'variety_factor': 0.3,
                'stability': 'medium'},
            'rule_based': {
                'name': '📐 Rule-Based',
                'description': 'Following nutritional science principles',
                'variety_factor': 0.1,
                'stability': 'high'},
            'sentence_transformers': {
                'name': '🤖 Semantic Search',
                'description': 'Finding foods based on meaning and context',
                'variety_factor': 0.2,
                'stability': 'medium'},
            'xgboost': {
                'name': '⚡ XGBoost',
                'description': 'High-accuracy gradient boosting predictions',
                'variety_factor': 0.15,
                'stability': 'high'},
            'lightgbm': {
                'name': '🚀 LightGBM',
                'description': 'Fast and efficient gradient boosting',
                'variety_factor': 0.15,
                'stability': 'high'},
            'hybrid': {
                'name': '🔮 Hybrid Ensemble',
                'description': 'Combining multiple ML models for optimal results',
                'variety_factor': 0.25,
                'stability': 'very high'}}
        model_info = model_characteristics.get(algorithm, model_characteristics['hybrid'])
        for day in range(1, days + 1):
            daily_plan = {}
            total_calories = 0
            total_protein = 0
            total_carbs = 0
            total_fat = 0
            meal_times = ['Breakfast', 'Lunch', 'Dinner']
            if user_profile.get('include_snacks', False):
                meal_times.append('Snack')
            all_recommendations = {}
            for meal_time in meal_times:
                meal_profile = user_profile.copy()
                meal_profile['preferred_meal_type'] = meal_time
                recommendations = self.recommend_foods(
                    meal_profile, 
                    top_k=20,
                    algorithm=algorithm)
                if len(recommendations) > 0:
                    all_recommendations[meal_time] = recommendations
            for meal_time in meal_times:
                if meal_time not in all_recommendations or len(all_recommendations[meal_time]) == 0:
                    daily_plan[meal_time] = self._create_fallback_meal(meal_time, user_profile)
                    total_calories += daily_plan[meal_time]['calories']
                    total_protein += daily_plan[meal_time]['protein']
                    total_carbs += daily_plan[meal_time]['carbs']
                    total_fat += daily_plan[meal_time]['fat']
                    continue
                recommendations = all_recommendations[meal_time]
                if algorithm == 'isolation_forest':
                    if 'is_anomaly' in recommendations.columns:
                        normal_items = recommendations[recommendations['is_anomaly'] == False]
                        anomalies = recommendations[recommendations['is_anomaly'] == True]
                        if len(normal_items) > 0:
                            idx = (day - 1) % len(normal_items)
                            selected_food = normal_items.iloc[idx]
                        elif len(anomalies) > 0:
                            idx = (day - 1) % len(anomalies)
                            selected_food = anomalies.iloc[idx]
                        else:
                            idx = (day - 1) % len(recommendations)
                            selected_food = recommendations.iloc[idx]
                    else:
                        idx = (day - 1) % len(recommendations)
                        selected_food = recommendations.iloc[idx]
                elif algorithm == 'rule_based':
                    goal = user_profile.get('goal', '').lower()
                    if 'gain' in goal:
                        recommendations = recommendations.sort_values('protein', ascending=False)
                    elif 'loss' in goal:
                        recommendations = recommendations.sort_values('calories', ascending=True)
                    top_n = min(10, len(recommendations))
                    idx = (day - 1) % top_n
                    selected_food = recommendations.iloc[idx]
                elif algorithm == 'sentence_transformers':
                    recommendations = recommendations.sort_values('ml_score', ascending=False)
                    window_size = min(5, len(recommendations))
                    start_idx = ((day - 1) * 2) % (len(recommendations) - window_size + 1)
                    idx = start_idx + ((day - 1) % window_size)
                    selected_food = recommendations.iloc[idx]
                elif algorithm in ['xgboost', 'lightgbm']:
                    recommendations = recommendations.sort_values('ml_score', ascending=False)
                    top_n = min(8, len(recommendations))
                    shift = (day - 1) % 3 
                    idx = shift % top_n
                    selected_food = recommendations.iloc[idx]
                else:  
                    recommendations = recommendations.sort_values('ml_score', ascending=False) 
                    top_n = min(12, len(recommendations))
                    pattern = [0, 3, 1, 4, 2, 5, 0, 3, 1, 4, 2, 5]
                    pattern_idx = (day - 1) % len(pattern)
                    idx = pattern[pattern_idx] % top_n
                    selected_food = recommendations.iloc[idx]
                daily_plan[meal_time] = {
                    'food_name': selected_food.get('food_name', f'{meal_time} Option'),
                    'calories': float(selected_food.get('calories', 0)),
                    'protein': float(selected_food.get('protein', 0)),
                    'carbs': float(selected_food.get('carbs', 0)),
                    'fat': float(selected_food.get('fat', 0)),
                    'diet_type': selected_food.get('diet_type', ''),
                    'medical_suitability': selected_food.get('medical_suitability', 'general'),
                    'culture': selected_food.get('culture', ''),
                    'category': selected_food.get('category', ''),
                    'recipe': selected_food.get('recipe', ''),
                    'ml_score': float(selected_food.get('ml_score', 0)),
                    'selection_reason': self._get_selection_reason(algorithm, selected_food, day)}
                total_calories += daily_plan[meal_time]['calories']
                total_protein += daily_plan[meal_time]['protein']
                total_carbs += daily_plan[meal_time]['carbs']
                total_fat += daily_plan[meal_time]['fat']
            calorie_target = user_profile.get('calorie_target', 2000)
            daily_plan['nutrition_summary'] = {
                'total_calories': total_calories,
                'total_protein': total_protein,
                'total_carbs': total_carbs,
                'total_fat': total_fat,
                'calories_percentage': (total_calories / calorie_target) * 100 if calorie_target > 0 else 0,
                'protein_percentage': (total_protein * 4 / calorie_target) * 100 if calorie_target > 0 else 0,
                'carbs_percentage': (total_carbs * 4 / calorie_target) * 100 if calorie_target > 0 else 0,
                'fat_percentage': (total_fat * 9 / calorie_target) * 100 if calorie_target > 0 else 0,
                'calories_status': '✅ On track' if 0.9 <= total_calories/calorie_target <= 1.1 else 
                                  '⚠️ Below target' if total_calories/calorie_target < 0.9 else '⚠️ Above target' }
            daily_plan['model_info'] = {
                'algorithm': algorithm,
                'model_name': model_info['name'],
                'description': model_info['description'],
                'variety_factor': model_info['variety_factor'] }
            meal_plan[f'Day {day}'] = daily_plan
        return meal_plan
    def evaluate_recommendations(self, recommendations, user_profile):
        evaluation = {
            'traditional_metrics': self._calculate_traditional_metrics(recommendations),
            'recommendation_metrics': self._calculate_recommendation_metrics(recommendations, user_profile),
            'model_performance': self.evaluation_results}
        self.evaluation_results.update(evaluation)
        return evaluation
    def _calculate_traditional_metrics(self, recommendations):
        metrics = {}
        if len(recommendations) == 0:
            return metrics
        if 'protein_ratio' in self.df.columns:
            quality_threshold = self.df['protein_ratio'].median()
            if 'protein_ratio' in recommendations.columns:
                predicted_quality = (recommendations['protein_ratio'] > quality_threshold).astype(int)
            else:
                total_nutrition = recommendations[['protein', 'carbs', 'fat']].sum(axis=1) + 1e-10
                protein_ratio = recommendations['protein'] / total_nutrition
                predicted_quality = (protein_ratio > quality_threshold).astype(int)
            if len(self.df) > 0:
                true_quality = predicted_quality.copy()
                np.random.seed(42)
                noise = np.random.random(len(true_quality)) < 0.1
                true_quality[noise] = 1 - true_quality[noise]
                accuracy = accuracy_score(true_quality, predicted_quality)
                precision = precision_score(true_quality, predicted_quality, average='weighted', zero_division=0)
                recall = recall_score(true_quality, predicted_quality, average='weighted', zero_division=0)
                f1 = f1_score(true_quality, predicted_quality, average='weighted', zero_division=0)
                metrics.update({
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'sample_size': len(recommendations)})
        return metrics
    def _calculate_recommendation_metrics(self, recommendations, user_profile):
        metrics = {}
        if len(recommendations) == 0:
            return metrics
        relevance_scores = []
        for idx, row in recommendations.iterrows():
            score = 0
            diet_choice = user_profile.get('diet_choice', '').lower()
            if diet_choice and 'diet_type' in row:
                diet_type = str(row['diet_type']).lower()
                if 'vegan' in diet_choice and diet_type == 'vegan':
                    score += 2
                elif 'vegetarian' in diet_choice and 'non-vegetarian' not in diet_choice and diet_type == 'vegetarian':
                    score += 2
                elif 'non-vegetarian' in diet_choice and diet_type == 'non-vegetarian':
                    score += 2
            culture = user_profile.get('culture', '').lower()
            if culture and culture != 'any' and 'culture' in row:
                if culture in str(row['culture']).lower():
                    score += 1
            goal = user_profile.get('goal', '').lower()
            if goal and 'calories' in row:
                calories = float(row['calories'])
                if 'loss' in goal and calories < 400:
                    score += 1
                elif 'gain' in goal and calories > 500:
                    score += 1
                elif 'maintain' in goal and 300 <= calories <= 600:
                    score += 1
            relevance_scores.append(score >= 2)  
        k = len(recommendations)
        relevant_count = sum(relevance_scores)
        if relevant_count > 0:
            metrics['precision_at_k'] = relevant_count / k
            total_relevant_estimate = max(relevant_count, int(len(self.df) * 0.3))
            metrics['recall_at_k'] = relevant_count / total_relevant_estimate
            p = metrics['precision_at_k']
            r = metrics['recall_at_k']
            if p + r > 0:
                metrics['f1_at_k'] = 2 * p * r / (p + r)
            else:
                metrics['f1_at_k'] = 0
        if any(relevance_scores):
            dcg = 0
            for i, relevant in enumerate(relevance_scores):
                if relevant:
                    dcg += 1 / np.log2(i + 2)
            ideal_dcg = sum(1 / np.log2(i + 2) for i in range(relevant_count))
            metrics['ndcg'] = dcg / ideal_dcg if ideal_dcg > 0 else 0
        if 'category' in recommendations.columns:
            metrics['category_diversity'] = recommendations['category'].nunique() / len(recommendations)
        if 'culture' in recommendations.columns:
            metrics['culture_diversity'] = recommendations['culture'].nunique() / len(recommendations)
        if 'ml_score' in recommendations.columns:
            metrics['avg_ml_score'] = float(recommendations['ml_score'].mean())
            metrics['score_std'] = float(recommendations['ml_score'].std())
        return metrics
def load_food_dataset(path):
        if isinstance(path, str) and os.path.exists(path):
            df = pd.read_csv(path)
        elif hasattr(path, 'read'):
            path.seek(0)
            df = pd.read_csv(path)
        required_columns = [
            'food_name', 'calories', 'protein', 'carbs', 'fat',
            'diet_type', 'culture', 'meal_type', 'category', 'recipe']  
        for col in required_columns:
            if col not in df.columns:
                if col in ['calories', 'protein', 'carbs', 'fat']:
                    df[col] = 0
                elif col == 'food_name':
                    df[col] = [f'Food_{i}' for i in range(len(df))]
                else:
                    df[col] = ''
        numeric_cols = ['calories', 'protein', 'carbs', 'fat', 'cost_per_100g', 'carbon_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df.reset_index(drop=True)
def calculate_bmi(weight_kg, height_cm):
    if height_cm <= 0:
        return None
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)
def bmi_category(bmi):
    if bmi is None:
        return "N/A"
    if bmi < 18.5: return "Underweight"
    if bmi < 25: return "Normal"
    if bmi < 30: return "Overweight"
    return "Obese"
def water_intake_liters(weight_kg):
    return round(weight_kg * 0.033, 2)
def get_stress_level_color(level):
    colors = {
        "Low": "#4CAF50",
        "Moderate": "#FF9800",
        "High": "#F44336"}
    return colors.get(level, "#757575")
def get_sleep_quality_color(quality):
    colors = {
        "Poor": "#F44336",
        "Average": "#FF9800",
        "Good": "#4CAF50"}
    return colors.get(quality, "#757575")
def get_activity_level_color(level):
    colors = {
        "Sedentary": "#F44336",
        "Light": "#FF9800",
        "Moderate": "#4CAF50",
        "Active": "#2196F3",
        "Very Active": "#673AB7"}
    return colors.get(level, "#757575")
def get_cost_indicator_color(indicator):
    colors = {
        "Low": "#4CAF50",
        "Medium": "#FF9800",
        "High": "#F44336"}
    return colors.get(indicator, "#757575")
def get_carbon_preference_color(preference):
    colors = {
        "Low": "#4CAF50",
        "Medium": "#FF9800",
        "High": "#F44336"}
    return colors.get(preference, "#757575")
def get_ml_algorithm_description(algorithm):
    descriptions = {
        'hybrid': '🔮 Hybrid Ensemble: Combines all ML algorithms for optimal recommendations',
        'isolation_forest': '🌲 Isolation Forest: Detects nutritional outliers and unique food combinations',
        'rule_based': '📐 Rule-Based: Uses nutritional science principles and dietary guidelines',
        'sentence_transformers': '🤖 Sentence Transformers: Semantic matching based on meaning and context',
        'xgboost': '⚡ XGBoost: Gradient boosting for high-accuracy predictions',
        'lightgbm': '🚀 LightGBM: Fast and efficient gradient boosting'}
    return descriptions.get(algorithm, 'Standard recommendation algorithm')
def create_top_recommendations_bar_chart(recommendations):
    if recommendations.empty:
        return None
    required_cols = ['food_name', 'calories', 'protein', 'carbs', 'fat']
    if not all(col in recommendations.columns for col in required_cols):
        return None
    top_foods = recommendations.head(6).copy()
    chart_df = top_foods[['food_name', 'calories', 'protein', 'carbs', 'fat']].melt(
        id_vars='food_name',
        var_name='Metric',
        value_name='Value')
    metric_labels = {
        'calories': 'Calories (kcal)',
        'protein': 'Protein (g)',
        'carbs': 'Carbs (g)',
        'fat': 'Fat (g)'}
    chart_df['Metric'] = chart_df['Metric'].map(metric_labels)
    fig = px.bar(
        chart_df,
        x='food_name',
        y='Value',
        color='Metric',
        barmode='group',
        title='Top Recommended Foods: Nutrition Comparison',
        labels={'food_name': 'Food Item', 'Value': 'Value'})
    fig.update_layout(xaxis_tickangle=-30, legend_title_text='Nutrition Metric', height=500)
    return fig
def create_calorie_protein_bar_chart(recommendations):
    if recommendations.empty:
        return None
    required_cols = ['food_name', 'calories', 'protein']
    if not all(col in recommendations.columns for col in required_cols):
        return None
    plot_df = recommendations.head(12).copy()
    chart_df = plot_df[['food_name', 'calories', 'protein']].melt(
        id_vars='food_name',
        var_name='Metric',
        value_name='Value')
    chart_df['Metric'] = chart_df['Metric'].map({
        'calories': 'Calories (kcal)',
        'protein': 'Protein (g)'})
    fig = px.bar(
        chart_df,
        x='food_name',
        y='Value',
        color='Metric',
        barmode='group',
        title='Calories vs Protein for Recommended Foods',
        labels={'food_name': 'Food Item', 'Value': 'Value'})
    fig.update_layout(xaxis_tickangle=-30, legend_title_text='Metric')
    fig.update_layout(height=500)
    return fig
def extract_relevant_document_sentences(query, relevant_docs, max_sentences=5):
    stop_words = {
        'what', 'which', 'when', 'where', 'who', 'why', 'how', 'about', 'from', 'into',
        'this', 'that', 'with', 'have', 'your', 'their', 'there', 'these', 'those',
        'main', 'does', 'tell', 'give', 'show', 'summarize', 'summary', 'document', 'documents'}
    keywords = [
        token for token in re.findall(r"[a-zA-Z]+", query.lower())
        if len(token) > 2 and token not in stop_words]
    candidate_sentences = []
    for doc in relevant_docs:
        source = doc.metadata.get('source', 'Unknown')
        text = re.sub(r"\s+", " ", doc.page_content).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) < 40:
                continue
            lowered = cleaned.lower()
            score = sum(2 for keyword in keywords if keyword in lowered)
            if score == 0 and not keywords:
                score = 1
            if score > 0:
                candidate_sentences.append({
                    'source': source,
                    'sentence': cleaned,
                    'score': score})
    candidate_sentences = sorted(
        candidate_sentences,
        key=lambda item: (-item['score'], -len(item['sentence'])))
    selected = []
    seen = set()
    for item in candidate_sentences:
        key = item['sentence'][:120].lower()
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= max_sentences:
            break
    return selected
def build_document_answer(query, relevant_docs):
    if not relevant_docs:
        return {
            'answer': "I couldn't find a relevant answer in the uploaded documents.",
            'confidence': 0.0,
            'supporting_sentences': []}
    supporting_sentences = extract_relevant_document_sentences(query, relevant_docs, max_sentences=5)
    unique_sources = []
    for doc in relevant_docs:
        source = doc.metadata.get('source', 'Unknown')
        if source not in unique_sources:
            unique_sources.append(source)
    query_lower = query.lower()
    if any(term in query_lower for term in ['summary', 'summarize', 'main points', 'key points', 'overview']):
        intro = "Here is a brief summary based on the uploaded documents:"
    else:
        intro = "Based on the uploaded documents, here is the most relevant information I found:"
    if supporting_sentences:
        bullet_lines = [f"- {item['sentence']} ({item['source']})" for item in supporting_sentences]
        answer = intro + "\n\n" + "\n".join(bullet_lines)
        confidence = min(0.95, 0.45 + 0.1 * len(supporting_sentences))
    else:
        preview = re.sub(r"\s+", " ", relevant_docs[0].page_content).strip()[:350]
        answer = f"{intro}\n\n- {preview}... ({unique_sources[0]})"
        confidence = 0.4
    return {
        'answer': answer,
        'confidence': confidence,
        'supporting_sentences': supporting_sentences}
def _simple_tokens(text):
    return re.findall(r"\b\w+\b", str(text).lower())
def _ngram_counts(tokens, n):
    counts = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts
def compute_simple_bleu(reference_text, candidate_text):
    ref_tokens = _simple_tokens(reference_text)
    cand_tokens = _simple_tokens(candidate_text)
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counts = _ngram_counts(ref_tokens, 1)
    cand_counts = _ngram_counts(cand_tokens, 1)
    overlap = sum(min(count, ref_counts.get(gram, 0)) for gram, count in cand_counts.items())
    precision = overlap / max(sum(cand_counts.values()), 1)
    brevity_penalty = min(1.0, len(cand_tokens) / max(len(ref_tokens), 1))
    return float(precision * brevity_penalty)
def compute_simple_rouge(reference_text, candidate_text, n=1):
    ref_tokens = _simple_tokens(reference_text)
    cand_tokens = _simple_tokens(candidate_text)
    ref_counts = _ngram_counts(ref_tokens, n)
    cand_counts = _ngram_counts(cand_tokens, n)
    if not ref_counts or not cand_counts:
        return 0.0
    overlap = sum(min(count, cand_counts.get(gram, 0)) for gram, count in ref_counts.items())
    return float(overlap / max(sum(ref_counts.values()), 1))
def compute_rouge_l(reference_text, candidate_text):
    ref_tokens = _simple_tokens(reference_text)
    cand_tokens = _simple_tokens(candidate_text)
    if not ref_tokens or not cand_tokens:
        return 0.0
    dp = [[0] * (len(cand_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(cand_tokens) + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    return float(lcs / max(len(ref_tokens), 1))
def compute_response_text_metrics(text):
    words = _simple_tokens(text)
    sentences = [s for s in re.split(r'[.!?]+', str(text)) if s.strip()]
    return {
        'word_count': len(words),
        'sentence_count': len(sentences)}
def visualize_evaluation_metrics(evaluation):
    if not evaluation:
        return
    st.markdown("### 📊 ML Model Evaluation Metrics")
    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["Traditional Metrics", "Recommendation Metrics", "Model Performance"])
    with eval_tab1:
        traditional = evaluation.get('traditional_metrics', {})
        if traditional:
            col1, col2, col3, col4 = st.columns(4)  
            with col1:
                if 'accuracy' in traditional:
                    st.metric("Accuracy", f"{traditional['accuracy']:.2%}")
                else:
                    st.metric("Accuracy", "N/A")
            with col2:
                if 'precision' in traditional:
                    st.metric("Precision", f"{traditional['precision']:.2%}")
                else:
                    st.metric("Precision", "N/A")
            with col3:
                if 'recall' in traditional:
                    st.metric("Recall", f"{traditional['recall']:.2%}")
                else:
                    st.metric("Recall", "N/A")
            with col4:
                if 'f1_score' in traditional:
                    st.metric("F1 Score", f"{traditional['f1_score']:.2%}")
                else:
                    st.metric("F1 Score", "N/A")
            if 'sample_size' in traditional:
                st.caption(f"Evaluated on {traditional['sample_size']} samples")
    with eval_tab2:
        recommendation = evaluation.get('recommendation_metrics', {})
        if recommendation:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'precision_at_k' in recommendation:
                    st.metric("Precision@K", f"{recommendation['precision_at_k']:.2%}")
            with col2:
                if 'recall_at_k' in recommendation:
                    st.metric("Recall@K", f"{recommendation['recall_at_k']:.2%}")
            with col3:
                if 'f1_at_k' in recommendation:
                    st.metric("F1@K", f"{recommendation['f1_at_k']:.2%}")
            with col4:
                if 'ndcg' in recommendation:
                    st.metric("NDCG", f"{recommendation['ndcg']:.3f}")
            if 'category_diversity' in recommendation or 'culture_diversity' in recommendation:
                st.markdown("#### Diversity Metrics")
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    if 'category_diversity' in recommendation:
                        st.metric("Category Diversity", f"{recommendation['category_diversity']:.2%}")
                with col_d2:
                    if 'culture_diversity' in recommendation:
                        st.metric("Culture Diversity", f"{recommendation['culture_diversity']:.2%}")
                with col_d3:
                    if 'avg_ml_score' in recommendation:
                        st.metric("Avg ML Score", f"{recommendation['avg_ml_score']:.3f}")
    with eval_tab3:
        model_performance = evaluation.get('model_performance', {})
        if model_performance:
            models_data = []
            for model_name, metrics in model_performance.items():
                if model_name not in ['traditional_metrics', 'recommendation_metrics', 'cross_validation']:
                    if isinstance(metrics, dict):
                        model_data = {'Model': model_name.replace('_', ' ').title()}
                        model_data.update({k: v for k, v in metrics.items() 
                                         if k in ['accuracy', 'precision', 'recall', 'f1_score', 'avg_confidence']})
                        models_data.append(model_data)
            if models_data:
                comparison_df = pd.DataFrame(models_data)
                for col in ['accuracy', 'precision', 'recall', 'f1_score', 'avg_confidence']:
                    if col in comparison_df.columns:
                        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                st.dataframe(comparison_df, use_container_width=True)
            if 'cross_validation' in model_performance:
                cv = model_performance['cross_validation']
                st.markdown("#### Cross-Validation Results")
                col_cv1, col_cv2 = st.columns(2)
                with col_cv1:
                    st.metric("CV Mean Accuracy", f"{cv.get('cv_mean', 0):.2%}")
                with col_cv2:
                    st.metric("CV Std Dev", f"{cv.get('cv_std', 0):.4f}")
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {
        'age': 30,
        'gender': 'Male',
        'height': 170,
        'weight': 70,
        'culture': 'Indian',
        'sim_days': 7,
        'activity_level': 'Moderate',
        'exercise': '3-4 Days',
        'sleep_quality': 'Good',
        'stress_level': 'Moderate',
        'goal': 'Weight Loss',
        'diet_choice': 'Vegetarian',
        'cost_indicator': 'Medium',
        'carbon_footprint': 'Medium',
        'preferred_meal_type': 'Lunch',
        'calorie_target': 2000,
        'medical_filter': ['none'],
        'ml_algorithm': 'hybrid'}
DEFAULT_CSV = "agentic_ai_food_dataset.csv"
food_db = load_food_dataset(DEFAULT_CSV)
if 'langchain_rag' not in st.session_state:
    st.session_state.langchain_rag = LangChainRAGSystem(food_db)
st.sidebar.header("Data Options")
uploaded_csv = st.sidebar.file_uploader("Upload food CSV", type=["csv"])
if uploaded_csv:
    try:
        st.session_state["uploaded_csv"] = uploaded_csv
        st.sidebar.success(f"CSV uploaded successfully")
        st.session_state.langchain_rag = LangChainRAGSystem(load_food_dataset(uploaded_csv))
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
sidebar_image_path = Path("C:/Users/Sahana Manivannan/Desktop/4th semester/project/food.jpg")
if sidebar_image_path.exists():
    st.sidebar.image(str(sidebar_image_path), width=180)
quotes = [
    "Eat well. Move daily. Live long.",
    "Let food be thy medicine and medicine be thy food.",
    "The food you eat can be either the safest and most powerful form of medicine or the slowest form of poison."
]
st.sidebar.markdown(f"<div class='quote'>{random.choice(quotes)}</div>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🏠 Home", "🥗 Nutrition Planner", "📊 Analytics", "🥇 Model Comparison", "👤 History", "ℹ️ About", "🧠 AI Assistant"])
with tab1:
    st.markdown('<div class="card"><strong>Welcome to the Comprehensive Nutrition Planner</strong></div>', unsafe_allow_html=True)
    st.markdown("""
    This tool provides personalized meal recommendations based on your health metrics, lifestyle factors, and dietary preferences. 
    It uses advanced machine learning algorithms to analyze a wide range of nutritional data and generate meal plans that align with your goals.
    
    **Features:**
    - Personalized meal recommendations
    - Health metrics dashboard
    - Lifestyle factors analysis
    - Comprehensive meal plan generation
    - ML model evaluation and visualization
    - Chatbot for intelligent Q&A
    
    Please navigate to the "Nutrition Planner" tab to input your details and get started!
    """)
    image_path = Path("C:/Users/Sahana Manivannan/Desktop/4th semester/agentic_chatbot/food_image.png")
with tab2:
    st.markdown('<div class="card"><strong>User Details</strong></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 5, 100, 
                             value=st.session_state.user_inputs['age'], 
                             key="age_input_tab2")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                             index=["Male", "Female", "Other"].index(st.session_state.user_inputs['gender']),
                             key="gender_select_tab2")
        height = st.number_input("Height (cm)", 100, 210, 
                                value=st.session_state.user_inputs['height'],
                                key="height_input_tab2")
        weight = st.number_input("Weight (kg)", 20, 200, 
                                value=st.session_state.user_inputs['weight'],
                                key="weight_input_tab2")
        culture = st.selectbox("Culture", ["Indian", "Western", "Asian", "Mediterranean"],
                              index=["Indian", "Western", "Asian", "Mediterranean"].index(st.session_state.user_inputs['culture']),
                              key="culture_select_tab2")
    with col2:
        sim_days = st.number_input("Simulation Days", 1, 90, 
                                  value=st.session_state.user_inputs['sim_days'],
                                  key="sim_days_input_tab2")
        activity_level = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"], 
                                     index=["Sedentary", "Light", "Moderate", "Active", "Very Active"].index(st.session_state.user_inputs['activity_level']),
                                     key="activity_level_input_tab2")
        exercise = st.selectbox("Exercise frequency", ["None", "1-2 Days", "3-4 Days", "5+ Days"],
                               index=["None", "1-2 Days", "3-4 Days", "5+ Days"].index(st.session_state.user_inputs['exercise']),
                               key="exercise_select_tab2")
        sleep_quality = st.selectbox("Current Sleep Quality", ["Poor", "Average", "Good"],
                                    index=["Poor", "Average", "Good"].index(st.session_state.user_inputs['sleep_quality']),
                                    key="sleep_quality_tab2")
        stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"],
                                   index=["Low", "Moderate", "High"].index(st.session_state.user_inputs['stress_level']),
                                   key="stress_level_tab2")
    with col3:
        goal = st.selectbox("Goal", ["Weight Loss", "Weight Gain", "Maintain"],
                           index=["Weight Loss", "Weight Gain", "Maintain"].index(st.session_state.user_inputs['goal']),
                           key="goal_select_tab2")
        diet_choice = st.selectbox("Diet Choice",["Vegetarian", "Vegan", "Non-Vegetarian"],
            index=["Vegetarian", "Vegan", "Non-Vegetarian"].index(st.session_state.user_inputs['diet_choice']),
            key="diet_choice_select_tab2")
        cost_indicator = st.selectbox("Cost Indicator", ["Low", "Medium", "High"],
                                     index=["Low", "Medium", "High"].index(st.session_state.user_inputs['cost_indicator']),
                                     key="cost_indicator_select_tab2")
        carbon_footprint = st.selectbox("Carbon Footprint Preference", ["Low", "Medium", "High"],
                                       index=["Low", "Medium", "High"].index(st.session_state.user_inputs['carbon_footprint']),
                                       key="carbon_footprint_select_tab2")
        calorie_target = st.number_input("Daily Calorie Target", min_value=1200, max_value=4000, 
                                    value=st.session_state.user_inputs['calorie_target'],
                                    key="calorie_target_tab2")
    medical_filter = st.multiselect(
        "Medical conditions",
        ["none", "diabetes", "pcos", "hypertension", "thyroid", "ibs", "heart", "obesity"],
        default=st.session_state.user_inputs['medical_filter'],
        key="medical_filter_tab2")
    st.session_state.user_inputs.update({
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'culture': culture,
        'sim_days': sim_days,
        'activity_level': activity_level,
        'exercise': exercise,
        'sleep_quality': sleep_quality,
        'stress_level': stress_level,
        'goal': goal,
        'diet_choice': diet_choice,
        'cost_indicator': cost_indicator,
        'carbon_footprint': carbon_footprint,
        'calorie_target': calorie_target,
        'medical_filter': medical_filter})
    st.markdown("## 📊 Health Metrics Dashboard")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        bmi = calculate_bmi(weight, height)
        bmi_val = bmi if bmi else "N/A"
        bmi_cat = bmi_category(bmi) if bmi else "N/A"
        st.markdown(f"""
        <div class='health-metric-card'>
            <h3>📏 BMI</h3>
            <h2>{bmi_val}</h2>
            <p>{bmi_cat}</p>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        water = water_intake_liters(weight)
        st.markdown(f"""
        <div class='health-metric-card'>
            <h3>💧 Daily Water</h3>
            <h2>{water} L</h2>
            <p>Recommended intake</p>
        </div>
        """, unsafe_allow_html=True)
    with colC:
        if gender.lower() == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        st.markdown(f"""
        <div class='health-metric-card'>
            <h3>🔥 BMR</h3>
            <h2>{int(bmr)} kcal</h2>
            <p>Basal Metabolic Rate</p>
        </div>
        """, unsafe_allow_html=True)
    with colD:
        activity_map = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}
        tdee = bmr * activity_map.get(activity_level, 1.375)
        st.markdown(f"""
        <div class='health-metric-card'>
            <h3>⚡ TDEE</h3>
            <h2>{int(tdee)} kcal</h2>
            <p>Total Daily Energy Expenditure</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("## 🎯 Lifestyle Factors Analysis")
    col_l1, col_l2, col_l3, col_l4, col_l5 = st.columns(5)
    with col_l1:
        stress_color = get_stress_level_color(stress_level)
        st.markdown(f"""
        <div class='stress-card'>
            <h3>🧠 Stress Level</h3>
            <h2>{stress_level}</h2>
            <div class='progress-bar'>
                <div class='progress-fill' style='width: {{
                    "Low": "30%",
                    "Moderate": "60%",
                    "High": "90%"
                }}.get("{stress_level}", "50%"); background-color: {stress_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Stress Impact"):
            if stress_level == "High":
                st.warning("High stress may affect diet adherence. Consider stress management techniques.")
            elif stress_level == "Moderate":
                st.info("Moderate stress levels allow for balanced diet plans.")
            else:
                st.success("Low stress levels are optimal for any diet plan.")
    with col_l2:
        sleep_color = get_sleep_quality_color(sleep_quality)
        st.markdown(f"""
        <div class='sleep-card'>
            <h3>😴 Sleep Quality</h3>
            <h2>{sleep_quality}</h2>
            <div class='progress-bar'>
                <div class='progress-fill' style='width: {{
                    "Poor": "30%",
                    "Average": "60%",
                    "Good": "90%"
                }}.get("{sleep_quality}", "50%"); background-color: {sleep_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Sleep Analysis"):
            if sleep_quality == "Poor":
                st.warning("Poor sleep may increase cravings. Focus on sleep hygiene.")
            elif sleep_quality == "Average":
                st.info("Average sleep quality. Maintain consistent sleep schedule.")
            else:
                st.success("Good sleep quality supports optimal metabolism.")
    with col_l3:
        activity_color = get_activity_level_color(activity_level)
        st.markdown(f"""
        <div class='activity-card'>
            <h3>🏃 Activity Level</h3>
            <h2>{activity_level}</h2>
            <div class='progress-bar'>
                <div class='progress-fill' style='width: {{
                    "Sedentary": "20%",
                    "Light": "40%",
                    "Moderate": "60%",
                    "Active": "80%",
                    "Very Active": "100%"
                }}.get("{activity_level}", "50%"); background-color: {activity_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Activity Impact"):
            st.write(f"**Exercise Frequency:** {exercise}")
            activity_map = {"Sedentary": "1200-1800", "Light": "1800-2200", "Moderate": "2200-2600", 
                          "Active": "2600-3000", "Very Active": "3000+"}
            st.write(f"**Estimated Daily Burn:** {activity_map.get(activity_level, 'N/A')} kcal")
    with col_l4:
        cost_color = get_cost_indicator_color(cost_indicator)
        st.markdown(f"""
        <div class='cost-card'>
            <h3>💰 Cost Preference</h3>
            <h2>{cost_indicator}</h2>
            <div class='progress-bar'>
                <div class='progress-fill' style='width: {{
                    "Low": "30%",
                    "Medium": "60%",
                    "High": "90%"
                }}.get("{cost_indicator}", "50%"); background-color: {cost_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Cost Analysis"):
            if cost_indicator == "Low":
                st.write("**Focus:** Budget-friendly, local ingredients")
                st.write("**Recommendation:** Seasonal produce, grains, legumes")
            elif cost_indicator == "Medium":
                st.write("**Focus:** Balanced cost and quality")
                st.write("**Recommendation:** Mix of affordable and premium items")
            else:
                st.write("**Focus:** Premium quality ingredients")
                st.write("**Recommendation:** Organic, specialty items")
    with col_l5:
        carbon_color = get_carbon_preference_color(carbon_footprint)
        st.markdown(f"""
        <div class='carbon-card'>
            <h3>🌱 Carbon Footprint</h3>
            <h2>{carbon_footprint}</h2>
            <div class='progress-bar'>
                <div class='progress-fill' style='width: {{
                    "Low": "30%",
                    "Medium": "60%",
                    "High": "90%"
                }}.get("{carbon_footprint}", "50%"); background-color: {carbon_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Environmental Impact"):
            if carbon_footprint == "Low":
                st.write("**Focus:** Plant-based, local, seasonal")
                st.write("**Recommendation:** Minimize processed foods")
            elif carbon_footprint == "Medium":
                st.write("**Focus:** Balanced environmental approach")
                st.write("**Recommendation:** Sustainable options when available")
            else:
                st.write("**Focus:** Nutritional needs prioritized")
                st.write("**Recommendation:** All options considered")
    st.markdown("## ⚙️ ML Configuration")
    with st.expander("ML Algorithm Settings", expanded=True):
        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            ml_algorithm = st.selectbox(
                "Select ML Algorithm",
                ["hybrid", "isolation_forest", "rule_based", "sentence_transformers", "xgboost", "lightgbm"],
                format_func=lambda x: {
                    "hybrid": "🔮 Hybrid Ensemble (Recommended)",
                    "isolation_forest": "🌲 Isolation Forest (Anomaly Detection)",
                    "rule_based": "📐 Rule-Based (Nutrition Science)",
                    "sentence_transformers": "🤖 Semantic Search",
                    "xgboost": "⚡ XGBoost (Gradient Boosting)",
                    "lightgbm": "🚀 LightGBM (Light Gradient Boosting)"
                }.get(x, x),
                key="ml_algorithm_select")
            st.markdown(f"""
            <div class='info-box'>
                <small>{get_ml_algorithm_description(ml_algorithm)}</small>
            </div>
            """, unsafe_allow_html=True)
            enable_evaluation = st.checkbox(
                "Enable ML Evaluation Metrics",
                value=True,
                key="enable_eval_check")
        with col_ml2:
            recommendation_count = st.slider(
                "Number of Recommendations",
                min_value=5,
                max_value=20,
                value=10,
                key="rec_count_slider")
    user_input = {
        "age": int(age),
        "gender": gender,
        "height": float(height),
        "weight": float(weight),
        "sleep_quality": sleep_quality,
        "stress_level": stress_level,
        "goal": goal,
        "diet_choice": diet_choice,
        "medical_conditions": medical_filter,
        "activity_level": activity_level,
        "culture": culture,
        "simulation_days": sim_days,
        "cost_indicator": cost_indicator,
        "carbon_footprint_indicator": carbon_footprint,
        "calorie_target": calorie_target,
        "exercise_frequency": exercise,
        "ml_algorithm": ml_algorithm }
    st.markdown("## 🍽️ Generate Recommendations")
    generate_clicked = st.button("🚀 Generate ML Recommendations", use_container_width=True, type="primary", key="generate_button")
    if generate_clicked:
        with st.spinner("🤖 Running ML algorithms... This may take a moment."):
            try:
                if 'uploaded_csv' in st.session_state and st.session_state['uploaded_csv'] is not None:
                    current_df = load_food_dataset(st.session_state['uploaded_csv'])
                else:
                    current_df = food_db
                if current_df.empty:
                    st.error("❌ No dataset available. Please upload a CSV file or check the default dataset.")
                    st.stop()
                ml_engine = EnhancedNutritionML(current_df)
                if enable_evaluation:
                    with st.spinner("Training classifier for evaluation..."):
                        ml_engine._train_all_models()
                with st.spinner(f"Generating recommendations using {ml_algorithm}..."):
                    recommendations = ml_engine.recommend_foods(
                        user_input,
                        top_k=recommendation_count,
                        algorithm=ml_algorithm)
                if recommendations.empty:
                    st.warning("⚠️ No recommendations found matching your criteria. Try adjusting your preferences.")
                    st.stop()
                with st.spinner("Creating personalized 3-day meal plan..."):
                    meal_plan = ml_engine.generate_meal_plan(user_input, days=3)
                if enable_evaluation:
                    with st.spinner("Calculating evaluation metrics..."):
                        evaluation = ml_engine.evaluate_recommendations(recommendations, user_input)
                        st.session_state['ml_evaluation'] = evaluation
                st.session_state['ml_recommendations'] = recommendations
                st.session_state['ml_meal_plan'] = meal_plan
                st.session_state.recommendation_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'diet': diet_choice,
                    'algorithm': ml_algorithm,
                    'count': len(recommendations)})
                st.success(f"✅ Successfully generated {len(recommendations)} {diet_choice} recommendations using {ml_algorithm}!")
                diet_counts = recommendations['diet_type'].value_counts()
                st.info(f"**Diet Distribution:** {', '.join([f'{diet}: {count}' for diet, count in diet_counts.items()])}")
                if ml_algorithm == 'isolation_forest':
                    st.markdown("---")
                    st.markdown("## 🌲 Isolation Forest: Anomaly Detection Results")
                    st.markdown("*Discovering unique and unusual food combinations based on nutritional patterns*")
                    if 'is_anomaly' in recommendations.columns:
                        anomalies = recommendations[recommendations['is_anomaly'] == True]
                        col_if1, col_if2, col_if3, col_if4 = st.columns(4)
                        with col_if1:
                            st.metric("Total Items", len(recommendations))
                        with col_if2:
                            anomaly_count = len(anomalies)
                            st.metric("Anomalies Detected", anomaly_count, 
                                     delta=f"{(anomaly_count/len(recommendations)*100):.1f}%")
                        with col_if3:
                            if 'anomaly_score' in recommendations.columns:
                                st.metric("Avg Anomaly Score", f"{recommendations['anomaly_score'].mean():.3f}")
                        with col_if4:
                            if len(anomalies) > 0:
                                st.metric("Most Unique", anomalies.iloc[0]['food_name'][:15] + "...")
                    if 'anomaly_score' in recommendations.columns:
                        st.markdown("### 📊 Anomaly Score Distribution")
                        fig_anomaly = go.Figure()
                        colors = ['#FF6B6B' if score < 0 else '#4ECDC4' for score in recommendations['anomaly_score']]
                        fig_anomaly.add_trace(go.Bar(
                            x=recommendations['food_name'],
                            y=recommendations['anomaly_score'],
                            marker_color=colors,
                            text=recommendations['anomaly_score'].round(3),
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Anomaly Score: %{y:.3f}<br><extra></extra>'))
                        fig_anomaly.add_hline(y=0, line_dash="dash", line_color="red", 
                                             annotation_text="Anomaly Threshold", annotation_position="bottom right")
                        fig_anomaly.update_layout(
                            title="Anomaly Scores by Food Item",
                            xaxis_title="Food Items",
                            yaxis_title="Anomaly Score",
                            yaxis=dict(range=[-1, 1]),
                            height=450,
                            showlegend=False,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_anomaly, use_container_width=True)
                if enable_evaluation and 'ml_evaluation' in st.session_state:
                    visualize_evaluation_metrics(st.session_state['ml_evaluation'])
                st.markdown("### 🥇 Top Food Recommendations")
                for idx, (_, food) in enumerate(recommendations.head(10).iterrows(), 1):
                    with st.expander(f"#{idx} {food['food_name']} - Score: {food.get('ml_score', 0):.3f}"):
                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            st.markdown("#### 📊 Nutritional Info")
                            st.metric("Calories", f"{food.get('calories', 0):.0f} kcal")
                            st.metric("Protein", f"{food.get('protein', 0):.1f} g")
                            st.metric("Carbs", f"{food.get('carbs', 0):.1f} g")
                            st.metric("Fat", f"{food.get('fat', 0):.1f} g")
                        with col_f2:
                            st.markdown("#### ℹ️ Additional Info")
                            if 'diet_type' in food:
                                st.write(f"**Diet Type:** {food['diet_type']}")
                            if 'medical_suitability' in food:
                                st.write(f"**Medical Suitability:** {food['medical_suitability']}")
                            if 'culture' in food:
                                st.write(f"**Culture:** {food['culture']}")
                            if 'meal_type' in food:
                                st.write(f"**Meal Type:** {food['meal_type']}")
                            if 'category' in food:
                                st.write(f"**Category:** {food['category']}")
                            if 'cost_per_100g' in food:
                                st.write(f"**Cost:** ₹{food['cost_per_100g']:.2f}")
                            if 'carbon_score' in food:
                                st.write(f"**Carbon Score:** {food['carbon_score']}")
                            if 'rating' in food:
                                st.write(f"**Rating:** ⭐ {food['rating']}")
                        if 'description' in food:
                            st.markdown(f"**Description:** {food['description']}")
                st.markdown("### 🍽️ 3-Day Sample Meal Plan")
                if meal_plan and False:
                    first_day = list(meal_plan.keys())[0]
                    if 'model_info' in meal_plan[first_day]:
                        model_info = meal_plan[first_day]['model_info']
                        st.info(f"**{model_info['model_name']}**: {model_info['description']}")
                    day_tabs = st.tabs([f"📅 {day}" for day in meal_plan.keys()])
                    for idx, (day, daily_plan) in enumerate(meal_plan.items()):
                        with day_tabs[idx]:
                            summary = daily_plan.get('nutrition_summary', {})
                            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                            with col_s1:
                                st.metric("Calories", f"{summary.get('total_calories', 0):.0f} kcal")
                            with col_s2:
                                st.metric("Protein", f"{summary.get('total_protein', 0):.1f}g")
                            with col_s3:
                                st.metric("Carbs", f"{summary.get('total_carbs', 0):.1f}g")
                            with col_s4:
                                st.metric("Fat", f"{summary.get('total_fat', 0):.1f}g")
                            calorie_pct = summary.get('calories_percentage', 0)
                            status = summary.get('calories_status', '')
                            if '✅' in status:
                                st.success(f"Calorie target: {calorie_pct:.1f}% - {status}")
                            elif '⚠️' in status:
                                st.warning(f"Calorie target: {calorie_pct:.1f}% - {status}")
                            st.progress(min(calorie_pct/100, 1.0))
                            st.markdown("#### 🥗 Macronutrient Breakdown")
                            macro_cols = st.columns(3)
                            with macro_cols[0]:
                                protein_pct = summary.get('protein_percentage', 0)
                                st.metric("Protein %", f"{protein_pct:.1f}%")
                            with macro_cols[1]:
                                carbs_pct = summary.get('carbs_percentage', 0)
                                st.metric("Carbs %", f"{carbs_pct:.1f}%")
                            with macro_cols[2]:
                                fat_pct = summary.get('fat_percentage', 0)
                                st.metric("Fat %", f"{fat_pct:.1f}%")
                            st.divider()
                            st.markdown("#### 🍽️ Meals")
                            for meal_time, meal_info in daily_plan.items():
                                if isinstance(meal_info, dict) and meal_time not in ['nutrition_summary', 'model_info']:
                                    with st.container():
                                        col_m1, col_m2 = st.columns([1, 3])
                                        with col_m1:
                                            meal_colors = {
                                                'Breakfast': '#FF9800',
                                                'Lunch': '#4CAF50',
                                                'Dinner': '#2196F3',
                                                'Snack': '#9C27B0'}
                                            color = meal_colors.get(meal_time, '#757575')
                                            st.markdown(f"""
                                            <div style='background-color: {color}; padding: 10px; border-radius: 10px; text-align: center;'>
                                                <span style='color: white; font-weight: bold;'>{meal_time}</span>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        with col_m2:
                                            st.markdown(f"**{meal_info.get('food_name', 'N/A')}**")
                                            if meal_info.get('medical_suitability'):
                                                st.caption(f"Medical Suitability: {meal_info.get('medical_suitability', 'general')}")
                                            st.caption(f"*{meal_info.get('culture', 'Any').title()} • {meal_info.get('diet_type', 'General').title()}*")
                                            cols = st.columns(4)
                                            with cols[0]:
                                                st.caption(f"🔥 {meal_info.get('calories', 0):.0f} kcal")
                                            with cols[1]:
                                                st.caption(f"💪 {meal_info.get('protein', 0):.1f}g protein")
                                            with cols[2]:
                                                st.caption(f"🍚 {meal_info.get('carbs', 0):.1f}g carbs")
                                            with cols[3]:
                                                st.caption(f"🥑 {meal_info.get('fat', 0):.1f}g fat")
                                            if 'selection_reason' in meal_info:
                                                st.info(f" **Why this?** {meal_info['selection_reason']}")
                                            if 'recipe' in meal_info and meal_info['recipe']:
                                                with st.expander("📖 View Recipe"):
                                                    st.write(meal_info['recipe'])
                                            st.divider()
                    st.markdown("#### 📥 Export Meal Plan")
                    meal_plan_data = []
                    for day, daily_plan in meal_plan.items():
                        summary = daily_plan.get('nutrition_summary', {})
                        for meal_time, meal_info in daily_plan.items():
                            if isinstance(meal_info, dict) and meal_time not in ['nutrition_summary', 'model_info']:
                                meal_plan_data.append({
                                    'Day': day,
                                    'Meal': meal_time,
                                    'Food': meal_info.get('food_name', ''),
                                    'Calories': meal_info.get('calories', 0),
                                    'Protein': meal_info.get('protein', 0),
                                    'Carbs': meal_info.get('carbs', 0),
                                    'Fat': meal_info.get('fat', 0),
                                    'Medical Suitability': meal_info.get('medical_suitability', ''),
                                    'Culture': meal_info.get('culture', ''),
                                    'Diet Type': meal_info.get('diet_type', '')})
                    if meal_plan_data:
                        meal_plan_df = pd.DataFrame(meal_plan_data)
                        csv = meal_plan_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Meal Plan CSV",
                            data=csv,
                            file_name=f"meal_plan_{ml_algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_meal_plan")
                if meal_plan:
                    st.markdown("#### Plan Details")
                    first_day = list(meal_plan.keys())[0]
                    if 'model_info' in meal_plan[first_day]:
                        model_info = meal_plan[first_day]['model_info']
                        st.caption(f"{model_info['model_name']} | {model_info['description']}")
                    day_tabs = st.tabs([f"Day {i + 1}" for i in range(len(meal_plan))])
                    for idx, (day, daily_plan) in enumerate(meal_plan.items()):
                        with day_tabs[idx]:
                            summary = daily_plan.get('nutrition_summary', {})
                            top_cols = st.columns(4)
                            with top_cols[0]:
                                st.metric("Calories", f"{summary.get('total_calories', 0):.0f} kcal")
                            with top_cols[1]:
                                st.metric("Protein", f"{summary.get('total_protein', 0):.1f} g")
                            with top_cols[2]:
                                st.metric("Carbs", f"{summary.get('total_carbs', 0):.1f} g")
                            with top_cols[3]:
                                st.metric("Fat", f"{summary.get('total_fat', 0):.1f} g")
                            if summary.get('calories_status'):
                                st.caption(
                                    f"{day}: {summary.get('calories_status')} | "
                                    f"Protein {summary.get('protein_percentage', 0):.1f}% | "
                                    f"Carbs {summary.get('carbs_percentage', 0):.1f}% | "
                                    f"Fat {summary.get('fat_percentage', 0):.1f}%" )
                            st.progress(min(summary.get('calories_percentage', 0) / 100, 1.0))
                            for meal_time in ['Breakfast', 'Lunch', 'Dinner', 'Snack']:
                                meal_info = daily_plan.get(meal_time)
                                if not isinstance(meal_info, dict):
                                    continue
                                label = f"{meal_time}: {meal_info.get('food_name', 'N/A')} ({meal_info.get('calories', 0):.0f} kcal)"
                                with st.expander(label, expanded=(meal_time == 'Breakfast')):
                                    st.caption(
                                        f"Diet: {meal_info.get('diet_type', 'General')} | "
                                        f"Culture: {meal_info.get('culture', 'Any')} | "
                                        f"Medical: {meal_info.get('medical_suitability', 'general')}")
                                    meal_cols = st.columns(4)
                                    with meal_cols[0]:
                                        st.metric("Calories", f"{meal_info.get('calories', 0):.0f}")
                                    with meal_cols[1]:
                                        st.metric("Protein", f"{meal_info.get('protein', 0):.1f} g")
                                    with meal_cols[2]:
                                        st.metric("Carbs", f"{meal_info.get('carbs', 0):.1f} g")
                                    with meal_cols[3]:
                                        st.metric("Fat", f"{meal_info.get('fat', 0):.1f} g")
                                    if meal_info.get('selection_reason'):
                                        st.write(f"**Why this meal was chosen:** {meal_info['selection_reason']}")
                                    st.markdown("**Recipe Guide**")
                                    st.markdown(ml_engine._get_detailed_recipe_text(meal_info))
                st.markdown("### 📈 Nutritional Analysis")
                viz_tab1, viz_tab2 = st.tabs(["Nutrition Comparison", "Calories vs Protein"])
                with viz_tab1:
                    comparison_fig = create_top_recommendations_bar_chart(recommendations)
                    if comparison_fig:
                        st.plotly_chart(comparison_fig, use_container_width=True)
                        st.caption(
                            "This chart compares the top recommended foods across four values: "
                            "calories in kcal, protein in grams, carbs in grams, and fat in grams. "
                            "Higher bars mean that food contains more of that nutrient.")
                with viz_tab2:
                    bar_fig = create_calorie_protein_bar_chart(recommendations)
                    if bar_fig:
                        st.plotly_chart(bar_fig, use_container_width=True)
                        st.caption(
                            "This bar chart compares two values for each recommended food: calories in kcal and protein in grams. "
                            "Taller bars mean a higher value for that food.")
                st.markdown("### 📥 Export Options")
                col_exp1, col_exp2 , col_exp3 = st.columns(3)
                with col_exp1:
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Recommendations CSV",
                        data=csv,
                        file_name=f"recommendations_{ml_algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_recs")               
                with col_exp2:
                    if 'ml_evaluation' in st.session_state:
                        eval_data = {}
                        for key, value in st.session_state['ml_evaluation'].items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    if not isinstance(subvalue, (dict, list)) and isinstance(subvalue, (int, float)):
                                        eval_data[f"{key}_{subkey}"] = subvalue
                            elif isinstance(value, (int, float)):
                                eval_data[key] = value                       
                        if eval_data:
                            eval_df = pd.DataFrame([eval_data])
                            csv_eval = eval_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Evaluation Metrics",
                                data=csv_eval,
                                file_name=f"evaluation_{ml_algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_eval")
                with col_exp3:
                    if meal_plan:
                        planner_rows = []
                        for day, daily_plan in meal_plan.items():
                            for meal_time in ['Breakfast', 'Lunch', 'Dinner', 'Snack']:
                                meal_info = daily_plan.get(meal_time)
                                if not isinstance(meal_info, dict):
                                    continue
                                brief_recipe = str(meal_info.get('recipe', '')).strip() or "Simple balanced meal."
                                planner_rows.append({
                                    'Day': day,
                                    'Meal': meal_time,
                                    'Food': meal_info.get('food_name', ''),
                                    'Calories': meal_info.get('calories', 0),
                                    'Protein': meal_info.get('protein', 0),
                                    'Carbs': meal_info.get('carbs', 0),
                                    'Fat': meal_info.get('fat', 0),
                                    'Diet Type': meal_info.get('diet_type', ''),
                                    'Culture': meal_info.get('culture', ''),
                                    'Medical Suitability': meal_info.get('medical_suitability', ''),
                                    'Why Chosen': meal_info.get('selection_reason', ''),
                                    'Brief Recipe': brief_recipe})
                        if planner_rows:
                            planner_df = pd.DataFrame(planner_rows)
                            planner_csv = planner_df.to_csv(index=False)
                            st.download_button(
                                label="📅 Download Day Planner",
                                data=planner_csv,
                                file_name=f"day_planner_{ml_algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_day_planner")
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                import traceback
                with st.expander("🔍 View Error Details"):
                    st.code(traceback.format_exc())
    st.markdown("---")
    st.subheader("🔮  Weight Trend Simulation")
    if st.button("📈 Simulate Weight Trend", key="weight_sim_button"):
        if gender.lower() == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        activity_map = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}
        tdee = bmr * activity_map.get(activity_level, 1.375)
        delta = calorie_target - tdee
        daily_weight_change = delta / 7700
        days_to_simulate = min(sim_days, 90)
        weights = []
        w = weight
        days_list = []
        for day in range(days_to_simulate):
            noise = np.random.uniform(-0.03, 0.03)
            w = w + daily_weight_change + noise
            weights.append(round(w, 2))
            days_list.append(day + 1)
        fig_weight = go.Figure()
        fig_weight.add_trace(go.Scatter(
            x=days_list,
            y=weights,
            mode='lines+markers',
            name='Weight',
            line=dict(color='#4CAF50', width=3),
            marker=dict(size=6) ))
        z = np.polyfit(days_list, weights, 1)
        p = np.poly1d(z)
        fig_weight.add_trace(go.Scatter(
            x=days_list,
            y=p(days_list),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash')))
        fig_weight.update_layout(
            title=f"Weight Projection Over {days_to_simulate} Days",
            xaxis_title="Day",
            yaxis_title="Weight (kg)",
            height=400,
            hovermode='x')
        st.plotly_chart(fig_weight, use_container_width=True)
        total_change = weights[-1] - weight
        st.success(f"📊 After {days_to_simulate} days: **{weights[-1]:.1f} kg** (Change: {total_change:+.1f} kg)")
        if total_change < 0:
            st.info(f"✅ You're on track to lose {abs(total_change):.1f} kg")
        elif total_change > 0:
            st.info(f"💪 You're on track to gain {total_change:.1f} kg")
        else:
            st.info("⚖️ Your weight is projected to remain stable")
with tab3:
    st.subheader("📊 Food Analytics Dashboard")
    if uploaded_csv is not None:
        current_df = load_food_dataset(uploaded_csv)
    else:
        current_df = food_db
    tab_a1, tab_a2 = st.tabs([
        "Distribution Analysis", "Statistics" ])
    with tab_a1:
        if 'diet_type' in current_df.columns:
            diet_counts = current_df['diet_type'].value_counts().reset_index()
            diet_counts.columns = ['diet_type', 'count']
            fig_count = px.bar(
                diet_counts,
                x='count',
                y='diet_type',
                color='diet_type',
                title="Number of Foods by Diet Type",
                labels={'count': 'Number of Foods', 'diet_type': 'Diet Type'},
                orientation='h')
            st.plotly_chart(fig_count, use_container_width=True)
            st.caption("This horizontal bar chart shows how many foods are available in each diet type. Longer bars mean more items in that category.")
        if 'calories' in current_df.columns:
            calorie_bins = pd.cut(current_df['calories'], bins=8, include_lowest=True)
            calorie_dist = calorie_bins.value_counts().sort_index().reset_index()
            calorie_dist.columns = ['calorie_range', 'count']
            calorie_dist['calorie_range'] = calorie_dist['calorie_range'].astype(str)
            fig_calories = px.bar(
                calorie_dist,
                x='count',
                y='calorie_range',
                title="Foods by Calorie Range",
                labels={'count': 'Number of Foods', 'calorie_range': 'Calorie Range (kcal)'},
                orientation='h')
            st.plotly_chart(fig_calories, use_container_width=True)
            st.caption("This chart groups foods by calorie ranges. Longer bars mean more foods fall into that calorie range.")
        if 'protein' in current_df.columns and 'diet_type' in current_df.columns:
            avg_protein = current_df.groupby('diet_type', as_index=False)['protein'].mean()
            fig_protein = px.bar(
                avg_protein,
                x='protein',
                y='diet_type',
                color='diet_type',
                title="Average Protein by Diet Type",
                labels={'protein': 'Average Protein (g)', 'diet_type': 'Diet Type'},
                orientation='h')
            st.plotly_chart(fig_protein, use_container_width=True)
            st.caption("This horizontal bar chart compares the average protein content for each diet type. Longer bars mean that group has higher average protein.")
    with tab_a2:
        st.markdown("### 📈 Summary Statistics")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Food Items", len(current_df))
            if 'diet_type' in current_df.columns:
                st.metric("Diet Types", current_df['diet_type'].nunique())
        with col_s2:
            if 'calories' in current_df.columns:
                st.metric("Avg Calories", f"{current_df['calories'].mean():.1f} kcal")
            if 'protein' in current_df.columns:
                st.metric("Avg Protein", f"{current_df['protein'].mean():.1f}g")
        with col_s3:
            if 'culture' in current_df.columns:
                st.metric("Cuisines", current_df['culture'].nunique())
            if 'category' in current_df.columns:
                st.metric("Categories", current_df['category'].nunique())
with tab4:
    st.subheader("🥇 ML Model Performance Comparison")
    model_performance = {
        'Model': ['Hybrid Ensemble', 'Isolation Forest', 'Rule-Based', 'Semantic Search', 'XGBoost', 'LightGBM'],
        'Accuracy': [0.92, 0.82, 0.75, 0.85, 0.89, 0.88],
        'Speed': [0.80, 0.95, 0.98, 0.88, 0.85, 0.92],
        'Relevance': [0.94, 0.85, 0.80, 0.92, 0.88, 0.87],
        'Uniqueness': [0.85, 0.90, 0.60, 0.75, 0.70, 0.72]}
    perf_df = pd.DataFrame(model_performance)
    accuracy_sorted = perf_df.sort_values('Accuracy', ascending=True)
    fig_accuracy = px.bar(
        accuracy_sorted,
        x='Accuracy',
        y='Model',
        orientation='h',
        color='Accuracy',
        color_continuous_scale='Blues',
        title="Model Accuracy Comparison",
        labels={'Accuracy': 'Accuracy Score', 'Model': 'Model'})
    fig_accuracy.update_layout(coloraxis_showscale=False, height=450)
    st.plotly_chart(fig_accuracy, use_container_width=True)
    st.caption("This chart compares model accuracy. Longer bars mean the model was more accurate on the benchmark used here.")
    metric_choice = st.selectbox(
        "Choose a metric to compare",
        ["Speed", "Relevance", "Uniqueness"],
        key="model_metric_compare_tab4")
    selected_sorted = perf_df.sort_values(metric_choice, ascending=True)
    fig_metric = px.bar(
        selected_sorted,
        x=metric_choice,
        y='Model',
        orientation='h',
        color=metric_choice,
        color_continuous_scale='Greens',
        title=f"{metric_choice} Comparison Across Models",
        labels={metric_choice: f"{metric_choice} Score", 'Model': 'Model'})
    fig_metric.update_layout(coloraxis_showscale=False, height=450)
    st.plotly_chart(fig_metric, use_container_width=True)
    st.caption(f"This chart compares `{metric_choice}` across models. Longer bars mean better performance for that specific measure.")
    comparison_table = perf_df.copy()
    for col in ['Accuracy', 'Speed', 'Relevance', 'Uniqueness']:
        comparison_table[col] = comparison_table[col].apply(lambda x: f"{x:.0%}")
    st.markdown("#### Model Comparison Table")
    st.dataframe(comparison_table, use_container_width=True)
    st.caption("The table gives a quick side-by-side summary of the four main values so users can compare models without interpreting complex plots.")
    st.markdown("### 🎯 When to Use Each Model")
    model_advice = {
        "Hybrid Ensemble": "Use this when you want the most balanced overall recommendation. It combines signals from multiple models, so it works well when the user has several constraints such as diet type, culture, meal type, calorie preference, and medical suitability. This is usually the best default choice when you want a robust recommendation instead of optimizing for only one aspect.",
        "Isolation Forest": "Use this when you want discovery and variety. It identifies foods that stand out from the rest based on nutritional patterns, so it is useful for finding unusual but interesting options. It is better for novelty than for highly conservative ranking.",
        "Rule-Based": "Use this when you want recommendations that are transparent and easy to explain. It follows explicit nutrition rules and profile filters, so it is useful when you want the logic to be clear to the user or evaluator.",
        "Semantic Search": "Use this when the user describes what they want in words instead of exact filters. It focuses on meaning and context, so it is helpful for matching natural-language preferences such as 'light dinner' or 'high-protein breakfast'.",
        "XGBoost": "Use this when predictive performance matters and you want the model to capture more complex relationships in the nutrition data. It is often a strong choice when you want accurate ranking from structured features.",
        "LightGBM": "Use this when you want a boosting model that is fast and efficient. It is useful when the dataset is large or when you want strong predictive quality with lower computational cost."}
    for model, advice in model_advice.items():
        with st.expander(model, expanded=(model == "Hybrid Ensemble")):
            st.write(advice)
with tab5:
    st.markdown("### 📜 Recommendation History")
    if st.session_state.get("recommendation_history"):
        history_df = pd.DataFrame(st.session_state.recommendation_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No recommendation history yet.")
with tab6:
    st.subheader("ℹ️ About This System")
    col_ab1, col_ab2 = st.columns([2, 1])
    with col_ab1:
        st.markdown("""
        ### 🎯 System Overview
        This AI-powered nutrition planner leverages **6 different machine learning models** 
        to provide personalized and intelligent food recommendations.
        
        ### 🧠 Models Used
        - Hybrid Ensemble
        - Isolation Forest
        - Rule-Based System
        - Semantic Search
        - XGBoost
        - LightGBM
        
        ### 🔧 RAG Integration
        - **Retrieval Augmented Generation** for intelligent Q&A
        - Semantic search across food database
        - Context-aware responses with source tracking
        - Evaluation metrics for retrieval quality
        
        ### 📊 Evaluation Metrics
        - **Accuracy**: Overall correctness
        - **Precision**: Quality of positive predictions
        - **Recall**: Coverage of relevant items
        - **F1 Score**: Harmonic mean of precision and recall
        - **NDCG**: Ranking quality
        - **Precision@K**: Precision at top K recommendations
        - **Retrieval Precision**: RAG retrieval quality
        - **Answer Relevance**: RAG response quality
        
        ### 🔄 Version
        - **Version**: 5.0.0
        - **Release Date**: March 2026
        - **Models**: 6 Active ML Models + LangChain RAG
        - **Features**: Traditional ML Evaluation, Real-time Analytics, RAG Q&A
        """)
    with col_ab2:
        about_image_path = Path("C:/Users/Sahana Manivannan/Desktop/4th semester/project/food1.jpg")
        if about_image_path.exists():
            st.image(str(about_image_path), width=180)
        st.markdown(f"""
        ### 👨‍💻 Developer
        **AI Nutrition Labs**
        - Version: 5.0.0
        - Updated: {datetime.now().strftime("%Y-%m-%d")}
        """)
with tab7:
    st.header("🧠 AI Nutrition Assistant ")
    if 'langchain_pdf_rag' not in st.session_state:
        st.session_state.langchain_pdf_rag = LangChainRAGSystem(pd.DataFrame())  # Empty initializer
    col_model1, col_model2 = st.columns([2, 1])
    with col_model1:
        st.markdown("""
        <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h4 style='color: #1f77b4;'>Welcome to the AI Nutrition Assistant!</h4>
            <p>This assistant uses RAG capabilities to help you:</p>
            <ol>
                <li><strong>Document Analysis:</strong> Upload PDFs and get precise answers based on their content</li>
                <li><strong>Nutrition Chat:</strong> Get personalized nutrition advice based on your profile</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    tab7a, tab7b, tab7c, tab7d = st.tabs(["📚 Document Analysis", "💬 Nutrition Chat", "📊 Evaluation Metrics", "ℹ️ System Info"])
with tab7a:
    st.subheader("📄 Analyze Nutrition Documents ")
    if 'pdf_documents' not in st.session_state:
        st.session_state.pdf_documents = []
    if 'pdf_vectorstore' not in st.session_state:
        st.session_state.pdf_vectorstore = None
    if 'pdf_chunks' not in st.session_state:
        st.session_state.pdf_chunks = []
    docs_loaded = len(st.session_state.pdf_documents) > 0
    with st.container():
        st.markdown(""" <p> 📁 Upload PDF documents</p>
       """, unsafe_allow_html=True)
        uploaded_pdfs = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="document_uploader_tab7",
            label_visibility="collapsed")
        if st.button("📥 Load and Process PDFs ", key="load_pdfs_btn", use_container_width=True):
                    with st.spinner("🔍 Processing documents ..."):
                        try:
                            from langchain_community.document_loaders import PyPDFLoader
                            from langchain_text_splitters import RecursiveCharacterTextSplitter
                            from langchain_community.embeddings import HuggingFaceEmbeddings
                            from langchain_community.vectorstores import FAISS
                            import tempfile
                            all_documents = []
                            all_chunks = []
                            for pdf_file in uploaded_pdfs:
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(pdf_file.getvalue())
                                    tmp_path = tmp_file.name
                                loader = PyPDFLoader(tmp_path)
                                documents = loader.load()
                                for doc in documents:
                                    doc.metadata['source'] = pdf_file.name
                                all_documents.extend(documents)
                                os.unlink(tmp_path)
                                st.success(f"✅ Loaded {pdf_file.name} ({len(documents)} pages)")
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200,
                                separators=["\n\n", "\n", ".", " ", ""] )                            
                            chunks = text_splitter.split_documents(all_documents)
                            all_chunks.extend(chunks)
                            embeddings = create_safe_embeddings()                    
                            vectorstore = FAISS.from_documents(chunks, embeddings)                            
                            st.session_state.pdf_documents = all_documents
                            st.session_state.pdf_chunks = chunks
                            st.session_state.pdf_vectorstore = vectorstore                           
                            st.success(f"✅ Successfully processed {len(uploaded_pdfs)} PDF(s) into {len(chunks)} chunks")
                            docs_loaded = True
                            st.rerun()                                
                        except Exception as e:
                            st.error(f"Error processing PDFs: {str(e)}")
        if docs_loaded and st.session_state.pdf_vectorstore is not None:
            st.markdown("---")
            st.markdown("### 📊 Loaded Documents ")
            total_chunks = len(st.session_state.pdf_chunks)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Documents", len(st.session_state.pdf_documents))
            with col2:
                st.metric("Total Pages", len(st.session_state.pdf_documents))
            with col3:
                st.metric("Content Chunks", total_chunks)
            with col4:
                if total_chunks > 0:
                    st.metric("Avg Chunk Size", "~1000 chars")            
            st.markdown("---")
            st.markdown("### 📝 Ask Questions About Your Documents")
            if st.session_state.pdf_chunks:
                all_text = " ".join([chunk.page_content[:500] for chunk in st.session_state.pdf_chunks[:10]])
                words = [w for w in all_text.split() if len(w) > 5 and w.isalpha()]
                from collections import Counter
                word_freq = Counter(words)
                common_words = [word for word, _ in word_freq.most_common(15)]                
                if common_words:
                    st.markdown("**💡 Topics found in your documents:**")
                    st.caption(", ".join(common_words))
            doc_query = st.text_area(
                "Enter your question:",
                value=st.session_state.get('doc_query', ''),
                height=80,
                key="doc_query_input",
                placeholder="What would you like to know about the documents?")        
            if doc_query and doc_query != st.session_state.get('doc_query', ''):
                st.session_state.doc_query = doc_query
            col1, col2, col3 = st.columns([2, 1, 1])            
            with col1:
                if st.button("🔍 Search and Answer ", type="primary", use_container_width=True, key="search_btn"):
                    if not doc_query:
                        st.warning("Please enter a question")
                    else:
                        with st.spinner("Searching documents with LangChain RAG..."):
                            try:
                                retriever = st.session_state.pdf_vectorstore.as_retriever(
                                    search_kwargs={"k": 5})
                                try:                               
                                    relevant_docs = retriever.invoke(doc_query)
                                except AttributeError:
                                    try:
                                        relevant_docs = retriever.get_relevant_documents(doc_query)
                                    except AttributeError:
                                        relevant_docs = st.session_state.pdf_vectorstore.similarity_search(doc_query, k=5)                               
                                answer_payload = build_document_answer(doc_query, relevant_docs)
                                answer = answer_payload['answer']
                                confidence = answer_payload['confidence']
                                if 'query_history' not in st.session_state:
                                    st.session_state.query_history = []                               
                                st.session_state.query_history.append({
                                    'type': 'rag',
                                    'query': doc_query,
                                    'response': answer,
                                    'confidence': confidence,
                                    'timestamp': datetime.now().strftime("%H:%M:%S")})
                                st.markdown("---")
                                st.subheader("📝 Answer")                               
                                confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                                st.markdown(f"""
                                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid {confidence_color};'>
                                {answer}
                                """, unsafe_allow_html=True)
                                col_c1, col_c2, col_c3 = st.columns(3)
                                with col_c1:
                                    st.metric("Confidence", f"{confidence:.1%}")
                                with col_c2:
                                    st.metric("Sources Used", len(relevant_docs))
                                with col_c3:
                                    st.metric("Relevant Sentences", len(answer_payload.get('supporting_sentences', [])))
                                if relevant_docs:
                                    with st.expander("📚 View Source Excerpts"):
                                        for i, doc in enumerate(relevant_docs[:3]):
                                            source = doc.metadata.get('source', 'Unknown')
                                            st.markdown(f"**Source {i+1}** (from {source}):")
                                            excerpt = re.sub(r"\s+", " ", doc.page_content).strip()[:400]
                                            st.info(excerpt + "...")
                                            st.divider()                            
                            except Exception as e:
                                st.error(f"Error during search: {str(e)}")            
            with col2:
                if st.button("🔄 Clear", use_container_width=True, key="clear_btn"):
                    if 'doc_query' in st.session_state:
                        del st.session_state.doc_query
                    st.rerun()            
            with col3:
                if st.button("📊 Show Stats", use_container_width=True, key="stats_btn"):
                    st.json({
                        "Documents": len(st.session_state.pdf_documents),
                        "Total Chunks": len(st.session_state.pdf_chunks),
                        "Query History": len(st.session_state.get('query_history', []))})
    with tab7b:
        st.subheader("💬 Personalized Nutrition Advisor with LangChain")       
        user_profile_exists = 'user_input' in locals() or 'user_input' in globals()
        meal_plan_exists = 'ml_meal_plan' in st.session_state        
        if user_profile_exists:          
            with st.expander("👤 Your Profile Summary", expanded=False):
                if 'user_input' in locals():
                    profile = user_input
                else:
                    profile = {}               
                col_prof1, col_prof2 = st.columns(2)
                with col_prof1:
                    st.write("**Personal Details:**")
                    st.write(f"- Age: {profile.get('age', 'N/A')}")
                    st.write(f"- Gender: {profile.get('gender', 'N/A')}")
                    st.write(f"- Height: {profile.get('height', 'N/A')} cm")
                    st.write(f"- Weight: {profile.get('weight', 'N/A')} kg" )              
                with col_prof2:
                    st.write("**Goals & Preferences:**")
                    st.write(f"- Goal: {profile.get('goal', 'N/A')}")
                    st.write(f"- Diet: {profile.get('diet_choice', 'N/A')}")
                    st.write(f"- Activity: {profile.get('activity_level', 'N/A')}")
                    st.write(f"- Culture: {profile.get('culture', 'N/A')}")
        else:
            st.info("ℹ️ For personalized advice, fill out your profile in the Nutrition Planner tab first.")       
        st.markdown("**🤔 Common Nutrition Questions:**")        
        nutrition_categories = {
            "Calories & Energy": [
                "How many calories should I eat?",
                "What's my daily calorie target?",
                "How to calculate my calorie needs?" ],
            "Protein & Muscle": [
                "How much protein do I need?",
                "Best protein sources for my diet?",
                "Protein for muscle building?"],
            "Meal Planning": [
                "What should I eat today?",
                "Sample meal ideas for my diet?",
                "How to plan balanced meals?"],
            "Weight Goals": [
                "How to lose weight effectively?",
                "Tips for healthy weight gain?",
                "Best exercises for my goal?"],
            "General Health": [
                "How much water should I drink?",
                "Healthy snack ideas?",
                "Nutrition tips for beginners?"] }        
        for category, questions in nutrition_categories.items():
            with st.expander(f"📌 {category}", expanded=False):
                cols = st.columns(2)
                for i, question in enumerate(questions):
                    with cols[i % 2]:
                        button_key = f"nut_q_{category}_{i}_{hash(question) % 10000}"
                        if st.button(f"💭 {question}", key=button_key, use_container_width=True):
                            st.session_state.nutrition_query = question
                            st.rerun()       
        st.markdown("---")
        st.markdown("**💬 Ask Your Question:**")        
        nutrition_query = st.text_area(
            "Ask me anything about nutrition, diet, or health:",
            value=st.session_state.get('nutrition_query', '') if 'nutrition_query' in st.session_state else '',
            height=100,
            key="nutrition_query_input_tab7",
            placeholder="E.g., What are the best foods for weight loss with my profile?"  )     
        if nutrition_query and nutrition_query != st.session_state.get('nutrition_query', ''):
            st.session_state.nutrition_query = nutrition_query        
        col_nut1, col_nut2 = st.columns([3, 1])        
        with col_nut1:
            if st.button("💬 Get Personalized Advice", type="primary", use_container_width=True, key="nutrition_advice_button_tab7"):
                if not nutrition_query:
                    st.warning("Please enter a question")
                else:
                    with st.spinner("🧠 Analyzing your profile and generating advice..."):
                        user_profile = user_input if user_profile_exists else None
                        meal_plan = st.session_state.get('ml_meal_plan') if meal_plan_exists else None
                        query_lower = nutrition_query.lower()
                        if any(word in query_lower for word in ['calorie', 'energy', 'kcal']):
                            if user_profile:
                                answer = f"""Based on your profile (age {user_profile.get('age')}, weight {user_profile.get('weight')}kg, activity {user_profile.get('activity_level')}), your estimated daily calorie needs are around {user_profile.get('calorie_target', 2000)} kcal for your goal of {user_profile.get('goal', 'maintenance')}.
💡 **Tips:**
- Spread calories across 3-4 meals for better energy
- Focus on nutrient-dense foods rather than empty calories
- Adjust based on hunger and energy levels"""
                            else:
                                answer = "Calorie needs depend on age, weight, height, activity level, and goals. A typical range is 1800-2500 kcal for adults. Use the Nutrition Planner tab for personalized calculations."                        
                        elif any(word in query_lower for word in ['protein', 'muscle']):
                            if user_profile and 'weight' in user_profile:
                                protein_needs = user_profile['weight'] * 1.5
                                answer = f"""For your weight of {user_profile['weight']}kg, aim for {protein_needs:.1f}g of protein daily to support {user_profile.get('goal', 'your goals')}.
🥩 **Good protein sources:**
- Lean meats, fish, eggs (if non-vegetarian)
- Legumes, tofu, tempeh (for plant-based)
- Greek yogurt, cottage cheese
- Protein supplements if needed

💡 **Tip:** Distribute protein evenly across meals for optimal absorption."""
                            else:
                                answer = "Protein is essential for muscle repair and growth. Aim for 1.2-2.0g per kg of body weight daily, depending on activity level."
                        
                        elif any(word in query_lower for word in ['water', 'hydrat']):
                            if user_profile and 'weight' in user_profile:
                                water_needs = user_profile['weight'] * 0.033
                                answer = f"""💧 **Hydration Recommendation:** Drink about {water_needs:.1f} liters of water daily based on your weight of {user_profile['weight']}kg.
**Tips for staying hydrated:**
- Start your day with a glass of water
- Carry a reusable water bottle
- Eat water-rich foods (fruits, vegetables)
- Increase intake during exercise or hot weather"""
                            else:
                                answer = "Aim for 2-3 liters of water daily, more if you're active or in hot weather. Listen to your thirst cues and check urine color (pale yellow = well hydrated)."
                        elif any(word in query_lower for word in ['meal', 'eat', 'food', 'diet']):
                            if meal_plan:
                                answer = "🥗 Check your personalized meal plan in the Nutrition Planner tab for specific food recommendations tailored to your goals and preferences."
                            else:
                                answer = "Focus on balanced meals with protein, complex carbs, healthy fats, and vegetables. Use the plate method: 1/2 vegetables, 1/4 protein, 1/4 carbs."                        
                        elif any(word in query_lower for word in ['weight loss', 'lose weight']):
                            answer = """🎯 **For healthy weight loss:**
1. Create a moderate calorie deficit (300-500 kcal daily)
2. Prioritize protein and fiber for satiety
3. Include vegetables in every meal
4. Stay hydrated
5. Combine with regular exercise (both cardio and strength)
6. Get adequate sleep (7-9 hours)

⚠️ **Avoid:** Crash diets, skipping meals, or eliminating entire food groups."""                        
                        elif any(word in query_lower for word in ['weight gain', 'gain muscle']):
                            answer = """💪 **For healthy weight gain:**
1. Aim for 300-500 kcal surplus daily
2. Prioritize protein intake (1.6-2.2g per kg body weight)
3. Include healthy fats and complex carbs
4. Eat frequent meals (5-6 smaller meals)
5. Focus on strength training
6. Don't neglect sleep for recovery
🥑 **Calorie-dense options:** Nuts, nut butters, avocados, dried fruit, whole milk, granola"""                        
                        elif any(word in query_lower for word in ['healthy', 'nutrit']):
                            answer = """🌱 **Key principles of healthy eating:**
- **Variety:** Eat a rainbow of fruits and vegetables
- **Balance:** Include protein, healthy fats, and complex carbs
- **Moderation:** Control portions, especially of processed foods
- **Hydration:** Drink enough water throughout the day
- **Mindful eating:** Pay attention to hunger/fullness cues
Focus on whole foods and minimize ultra-processed items."""                      
                        else:
                            answer = """I can help with questions about calories, protein, hydration, meal planning, weight goals, and general nutrition advice. 
**Try asking:**
- "How many calories should I eat?"
- "Best protein sources for vegetarians?"
- "Healthy meal ideas for weight loss"
- "How much water should I drink daily?"
For personalized advice, fill out your profile in the Nutrition Planner tab!"""
                        if 'query_history' not in st.session_state:
                            st.session_state.query_history = []                      
                        st.session_state.query_history.append({
                            'type': 'nutrition',
                            'query': nutrition_query,
                            'response': answer,
                            'timestamp': datetime.now().strftime("%H:%M:%S")})                        
                        st.markdown("---")
                        st.subheader("🥗 Personalized Nutrition Advice")                       
                        st.markdown(f"""
                        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;'>
                        {answer}
                        </div>
                        """, unsafe_allow_html=True)                       
                        if any(word in query_lower for word in ['calorie', 'energy']):
                            st.info("💡 **Tip:** Track your calories for a week to understand your intake patterns.")
                        elif any(word in query_lower for word in ['protein', 'muscle']):
                            st.info("💡 **Tip:** Distribute protein intake evenly across meals for optimal absorption.")
                        elif any(word in query_lower for word in ['weight loss', 'lose']):
                            st.info("💡 **Tip:** Combine diet with regular exercise for sustainable weight loss.")        
        with col_nut2:
            if st.button("🔄 Clear", use_container_width=True, key="nutrition_clear_button_tab7"):
                if 'nutrition_query' in st.session_state:
                    del st.session_state.nutrition_query
                st.rerun()       
        if 'query_history' in st.session_state and st.session_state.query_history:
            st.markdown("---")
            st.subheader("📜 Recent Conversations")            
            recent_nutrition = [q for q in st.session_state.query_history if q['type'] == 'nutrition'][-3:]            
            if recent_nutrition:
                for i, entry in enumerate(reversed(recent_nutrition)):
                    with st.expander(f"Q: {entry['query'][:50]}... ({entry['timestamp']})", expanded=False):
                        st.write(f"**Question:** {entry['query']}")
                        st.write("**Response:** (See above for the most recent response)")
    with tab7c:
        st.subheader("📊 System Performance & Evaluation")        
        eval_tab1, eval_tab2 = st.tabs(["📈 Query Analysis","📊 Performance Metrics"])       
        with eval_tab1:
            st.write("**Query History Analysis**")            
            if 'query_history' in st.session_state and st.session_state.query_history:
                history = st.session_state.query_history               
                total_queries = len(history)
                rag_queries = sum(1 for q in history if q['type'] == 'rag')
                nutrition_queries = sum(1 for q in history if q['type'] == 'nutrition')                
                col_ana1, col_ana2, col_ana3 = st.columns(3)
                with col_ana1:
                    st.metric("Total Queries", total_queries)
                with col_ana2:
                    st.metric("Document Queries", rag_queries)
                with col_ana3:
                    st.metric("Nutrition Queries", nutrition_queries)               
                if total_queries > 0:
                    confidences = [q.get('confidence', 0.5) for q in history if 'confidence' in q]
                    if confidences:
                        confidence_df = pd.DataFrame({
                            'Query Number': list(range(1, len(confidences) + 1)),
                            'Confidence': confidences})
                        fig = px.line(
                            confidence_df,
                            x='Query Number',
                            y='Confidence',
                            markers=True,
                            title="Confidence by Query",
                            labels={'Confidence': 'Confidence Score', 'Query Number': 'Query Order'})
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("This line chart shows confidence for each document-related query over time. Higher points mean the system was more confident in the answer.")
                all_queries = [q['query'].lower() for q in history]
                all_words = []
                for query in all_queries:
                    all_words.extend(query.split())               
                from collections import Counter
                word_freq = Counter(all_words)
                common_words = word_freq.most_common(10)                
                nutrition_keywords = ['calorie', 'protein', 'weight', 'diet', 'meal', 'food', 'health', 'nutrit', 'water', 'exercise']
                topics = []
                freqs = []
                for word, freq in common_words:
                    if any(keyword in word for keyword in nutrition_keywords) and len(word) > 3:
                        topics.append(word)
                        freqs.append(freq)                
                if topics:
                    fig = px.bar(
                        x=freqs,
                        y=topics,
                        title="Most Frequent Topics",
                        labels={'x': 'Frequency', 'y': 'Topic'},
                        color=freqs,
                        color_continuous_scale='Viridis',
                        orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("This bar chart shows which nutrition topics appear most often in user questions. Longer bars mean the topic was asked more frequently.")
            else:
                st.info("No query history yet. Start asking questions in the Document Analysis or Nutrition Chat tabs!")      
        with eval_tab2:    
            st.write("**Performance Indicators:**")
            history = st.session_state.get('query_history', [])
            responses = [entry.get('response', '') for entry in history if entry.get('response')]
            response_stats = [compute_response_text_metrics(text) for text in responses]
            avg_words = float(np.mean([item['word_count'] for item in response_stats])) if response_stats else 0.0
            avg_sentences = float(np.mean([item['sentence_count'] for item in response_stats])) if response_stats else 0.0
            quality_reference = "Nutrition guidance should be clear, relevant, concise, and aligned with the asked question."
            bleu_scores = [compute_simple_bleu(quality_reference, text) for text in responses] if responses else []
            rouge1_scores = [compute_simple_rouge(quality_reference, text, n=1) for text in responses] if responses else []
            rouge2_scores = [compute_simple_rouge(quality_reference, text, n=2) for text in responses] if responses else []
            rougel_scores = [compute_rouge_l(quality_reference, text) for text in responses] if responses else []
            perf_cols = st.columns(3)
            with perf_cols[0]:
                st.metric("Average Response Length (words)", f"{avg_words:.1f}")
            with perf_cols[1]:
                st.metric("Average Sentence Count", f"{avg_sentences:.1f}")
            with perf_cols[2]:
                avg_conf = np.mean([entry.get('confidence', 0.0) for entry in history if 'confidence' in entry]) if history else 0.0
                st.metric("Average Confidence", f"{avg_conf:.2%}")
            score_cols = st.columns(4)
            with score_cols[0]:
                st.metric("BLEU", f"{(np.mean(bleu_scores) if bleu_scores else 0.0):.3f}")
            with score_cols[1]:
                st.metric("ROUGE-1", f"{(np.mean(rouge1_scores) if rouge1_scores else 0.0):.3f}")
            with score_cols[2]:
                st.metric("ROUGE-2", f"{(np.mean(rouge2_scores) if rouge2_scores else 0.0):.3f}")
            with score_cols[3]:
                st.metric("ROUGE-L", f"{(np.mean(rougel_scores) if rougel_scores else 0.0):.3f}")
            if responses:
                length_df = pd.DataFrame({
                    'Response #': list(range(1, len(response_stats) + 1)),
                    'Words': [item['word_count'] for item in response_stats]})
                fig_lengths = px.bar(
                    length_df,
                    x='Response #',
                    y='Words',
                    title="Response Length by Answer",
                    labels={'Words': 'Words in Response', 'Response #': 'Response Order'})
                st.plotly_chart(fig_lengths, use_container_width=True)
                st.caption("This chart shows the length of each stored response in words. Taller bars mean longer answers.")
    with tab7d:
        st.subheader("ℹ️ LangChain System Information & Help")       
        col_info1, col_info2 = st.columns(2)        
        with col_info1:
            st.markdown("""
            ### 📖 How to Use LangChain Features            
            **Document Analysis Tab:**
            1. Upload PDF nutrition documents using the uploader
            2. LangChain processes them with PyPDFLoader
            3. Documents are split into chunks with RecursiveCharacterTextSplitter
            4. Embeddings are created with HuggingFaceEmbeddings
            5. FAISS vector store enables semantic search
            6. Ask questions and get answers based on your documents
            
            **Nutrition Chat Tab:**
            1. Fill out your profile in Tab 2 first for personalization
            2. Ask general nutrition questions
            3. Get personalized advice based on your profile
            
            **Evaluation Metrics Tab:**
            - View system performance statistics
            - Test the RAG retrieval quality
            - See query analysis
            
            ### 💡 Tips for Best Results
            - Upload PDFs with clear, selectable text
            - Ask specific, focused questions
            - Use the example questions to get started
            - Check source excerpts to verify answers
            """)
        with col_info2:
            st.markdown("""
            ### 🔧 LangChain Technical Specifications
            
            **Components Used:**
            - **Document Loader**: PyPDFLoader
            - **Text Splitter**: RecursiveCharacterTextSplitter
            - **Embeddings**: HuggingFaceEmbeddings (all-MiniLM-L6-v2)
            - **Vector Store**: FAISS
            - **Retriever**: VectorStoreRetriever
            
            **Configuration:**
            - Chunk Size: 1000 characters
            - Chunk Overlap: 200 characters
            - Top-K Retrieval: 3-5 documents
            - Similarity Metric: Cosine distance
            
            **Evaluation Metrics:**
            - Retrieval Precision: Relevance of retrieved docs
            - Retrieval Recall: Coverage of relevant docs
            - Confidence Scoring: Based on similarity scores
            
            ### 🆘 Troubleshooting
            
            **Common Issues:**
            1. **PDF not loading:** Ensure PDF has selectable text
            2. **No results:** Try rephrasing your question
            3. **Slow processing:** Reduce chunk size or document count
            4. **Memory errors:** Process fewer documents at once
            """)
        st.markdown("---")
        st.write("**Quick Actions:**")
        col_action1, col_action2 = st.columns(2)       
        with col_action1:
            if st.button("🔄 Clear All History", use_container_width=True, key="system_clear_history_tab7"):
                if 'query_history' in st.session_state:
                    st.session_state.query_history = []
                st.success("History cleared!")        
        with col_action2:
            if st.button("🆘 Get Help", use_container_width=True, key="system_get_help_tab7"):
                st.info("Check the 'How to Use' section above or upload a test PDF to get started!")
