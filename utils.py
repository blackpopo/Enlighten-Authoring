import requests
import os
from time import sleep
import tiktoken
current_dir = os.path.abspath(os.path.dirname(__file__))
data_folder = os.path.join(current_dir, 'database')
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import plotly.graph_objects as go
import networkx as nx
import openai
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import community as community_louvain # python-louvain packageをインストールする必要があるわ

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
DEEPL_API_KEY = st.secrets['DEEPL_API_KEY']
SEMANTICSCHOLAR_API_KEY = st.secrets['SEMANTICSCHOLAR_API_KEY']
AZURE_OPENAI_API_KEY = st.secrets['AZURE_OPENAI_KEY']
AZURE_OPENAI_ENDPOINT = st.secrets['AZURE_OPENAI_ENDPOINT']
    
# openai.api_key = OPENAI_API_KEY
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = 'azure'
openai.api_version = '2023-05-15'


def tiktoken_setup(offset = 8):
    gpt_35_tiktoken = tiktoken.encoding_for_model("gpt-3.5-turbo")
    gpt_35_16k_tiktoken = tiktoken.encoding_for_model("gpt-4-32k")
    gpt_4_tiktoken = tiktoken.encoding_for_model("gpt-4")
    gpt_4_32k_tiktoken = tiktoken.encoding_for_model("gpt-4-32k")

    tiktoken_dict = {
        "gpt-3.5-turbo": (gpt_35_tiktoken, 4097 - offset),
        "gpt-3.5-turbo-16k": (gpt_35_16k_tiktoken, 16385 - offset),
        "gpt-4": (gpt_4_tiktoken, 8192 - offset),
        "gpt-4-32k": (gpt_4_32k_tiktoken, 32768 - offset),
    }
    return tiktoken_dict

tiktoken_dict = tiktoken_setup()

def get_azure_gpt_response(system_input, model_name='gpt-4-32k'):
    response = openai.ChatCompletion.create(
        engine=model_name,
        messages=[
          {"role": "system", "content": system_input},
        ],
    )

    return response.choices[0]["message"]["content"].strip()

#クエリにもとづいて、Semantic Scholar から論文を offset を始めとして、 limit 分取得する。ただし、完全一致の論文のみが送信される。
def suggest_paper_completions(query):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/autocomplete"
    params = {"query": query[:100]}  # 最初の100文字だけを使用
    headers = {"x-api-key": SEMANTICSCHOLAR_API_KEY}
    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status {response.status_code}")
        return None


#クエリにもとづいて、Semantic Scholar から論文を offset を始めとして、 limit 分取得する。
def _get_papers(query, year, offset, limit, fields):
    base_url = "http://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "offset": offset, "limit": limit, "fields": fields, 'year': year}
    headers = {"x-api-key": SEMANTICSCHOLAR_API_KEY}  # API keyをヘッダーに追加

    retries = 10  # リトライする回数
    while retries > 0:
        try:
            response = requests.get(base_url, params=params, headers=headers)
            if response.status_code == 200:
              return response.json()
            elif response.status_code == 504:
              print(f"Request timed out, retrying... {retries} attempts left")
              retries -= 1
              sleep(1)  # タイムアウトした場合、少し待つ
            else:
              print(f"Request failed with status {response.status_code}")
              print(f"Request error message {response.reason}")
              print(f"Request content {response.content}")
              return None
        except Exception as e:
            print(f"ERROR HAPPENED AS {e}")
            return None

    print("All retries failed")
    return None

def get_papers(query_text, year, offset = 0, limit = 100, total_limit = 1000):
  papers = []
  # fields = "paperId,title,abstract,year,authors,journal,citationCount,citationStyles,referenceCount,references,references.title,references.abstract,references.year,references.authors,references.citationCount,references.referenceCount,references.citationStyles"
  fields = "paperId,title,abstract,year,authors,journal,citationCount,citationStyles,referenceCount,references,references.title,references.abstract,references.year,references.authors,references.citationCount,references.referenceCount,references.citationStyles,references.journal"

  # 最初の結果セットを取得
  # _get_papersはエラーが発生した場合 None を返す
  result = _get_papers(query_text, year, offset, limit, fields=fields)
  if not result or result['total'] == 0:
    return [], 0


  papers.extend(result['data'])
  print(f"Total results is {result['total']}.")
  print(f"{min(round(limit / min(total_limit, result['total']) * 100, 1), 100)}%")

  # 1000件未満なら全件取得
  while len(papers) < min(total_limit, result['total']):
      offset += limit
      result = _get_papers(query_text, year, offset, limit, fields=fields)
      if not result:
          if len(papers) > 0:
              break
          else:
            return [], 0
      print(f"{min(round(min(total_limit, offset + limit) / min(total_limit, result['total']) * 100, 1), 100)}%")
      papers.extend(result['data'])
  return papers, result['total']

#Reference を含めた正規化された論文一覧
def get_all_papers_from_references(papers):
    all_papers = papers.copy()
    all_papers.drop(columns=['references'], inplace=True)

    # 空のDataFrameを作成して、reference情報を格納する
    reference_df_list = []
    for index, row in papers.iterrows():
        references = row['references']
        if references:
            for reference in references:
                ref_dict = {'paperId': row['paperId']}
                ref_dict.update(reference)
                reference_df_list.append(ref_dict)

    reference_df = pd.DataFrame(reference_df_list)

    # 同じ 'paperId' を持つ行を合成
    all_papers.set_index('paperId', inplace=True)
    reference_df.set_index('paperId', inplace=True)
    all_papers = all_papers.combine_first(reference_df)

    # 重複する行を削除
    all_papers.reset_index(inplace=True)
    all_papers.drop_duplicates(subset=['paperId'], inplace=True)
    # all_papers.set_index('paperId', inplace=True)

    if 'Unnamed: 0' in all_papers.columns:
        all_papers.drop(columns=['Unnamed: 0'], inplace=True)
    if 'embedding' in all_papers.columns:
        all_papers.drop(columns=['embedding'], inplace=True)

    return all_papers
#論文 id のリストにもとづいて、Semantic Scholar から論文を offset を始めとして、 limit 分取得する。
def _get_papers_from_ids(paper_ids, fields):
    base_url = 'https://api.semanticscholar.org/graph/v1/paper/batch'
    headers = {"x-api-key": SEMANTICSCHOLAR_API_KEY}  # API keyをヘッダーに追加
    # fields = "paperId,title,abstract,year,authors,citationCount,referenceCount,references,embedding"
    retries = 10  # リトライする回数
    while retries > 0:
        try:
          response = requests.post(
              base_url,
              headers=headers,  # ヘッダーをリクエストに追加
              params={'fields': fields},
              json={"ids": paper_ids}
          )
          if response.status_code == 200:
              return response.json()
          elif response.status_code == 504:
              print(f"Request timed out, retrying... {retries} attempts left")
              retries -= 1
              sleep(5)  # タイムアウトした場合、少し待つ
          else:
              print(f"Request failed with status {response.status_code}")
              print(f"Request error message {response.reason}")
              print(f"Request content {response.content}")
              retries -= 1
              return None
        except Exception as e:
            print(f"ERROR HAPPENED AS {e}")
            return None
    return None


def get_papers_from_ids(paper_ids, offset=0, limit=100):
    total_results = []
    total_dict = {}
    fields = "paperId,title,abstract,year,authors,journal,citationCount,citationStyles,referenceCount,references,references.title,references.abstract,references.year,references.authors,references.citationCount,references.referenceCount,references.citationStyles,references.journal"

    # 修正した進捗表示
    total_len = len(paper_ids)

    # 修正したループ条件
    while offset < total_len:
        progress = round((offset + limit / total_len) * 100, 1)
        print(f"{min(progress, 100)}%")

        # ここで_get_papers_from_ids関数が呼び出されると仮定
        result = _get_papers_from_ids(paper_ids[offset: offset + limit], fields=fields)

        if result == None or len(result) == 0:
            break

        total_results.extend(result)
        offset += limit

    if len(total_results) == 0:
        return None, None

    for result_dict in total_results:
        for key, value in result_dict.items():
            total_dict.setdefault(key, []).append(value)

    dict_value_size = len(next(iter(total_dict.values()), []))
    is_same_size = all(len(v) == dict_value_size for v in total_dict.values())

    if not is_same_size:
        raise ValueError('Invalid value size!')

    return total_results, total_dict


def safe_filename(filename):
    """
    ファイル名を安全な形式に変換する。

    Parameters:
        filename (str): 元のファイル名

    Returns:
        str: 安全なファイル名
    """
    # 不要な文字を取り除く。ここではアルファベット、数字、アンダースコア、ハイフン、スペース以外のすべての文字を取り除きます。
    safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', filename)

    # スペースをアンダースコアに変換する（オプション）
    safe_name = safe_name.replace(' ', '_')

    return safe_name

def encode_to_filename(s):
    return s.replace(" ", "_").replace("\"", "__dq__")


def safe_literal_eval(val):
    # val が nan でない場合のみ、safe_literal_eval を適用
    if pd.notna(val):
        try:
            return ast.literal_eval(val)
        except Exception as e:
            # ここでエラーを処理
            print(f"Error in literal_eval: {e}")
            return val  # またはエラー処理に応じて別の値を返す
    return val  # nan の場合はそのまま返す

def decode_from_filename(filename):
    return filename.replace(".csv", "").replace('__dq__', '\"').replace("_", " ")

def save_papers_dataframe(df, query_text):
    encoded_query_text_without_extension = encode_to_filename(query_text)
    file_name = safe_filename(encoded_query_text_without_extension)
    df.to_csv(os.path.join(data_folder, f'{file_name}.csv'), encoding='utf-8')


def load_papers_dataframe(query_text, literal_evals = ['references', 'authors', 'citationStyles']):
    encoded_query_text_without_extension = encode_to_filename(query_text)
    file_name = safe_filename(encoded_query_text_without_extension)
    papers = pd.read_csv(os.path.join(data_folder, f'{file_name}.csv'))
    papers.reset_index(drop=True, inplace=True)

    for literal_eval_value in literal_evals:
        papers[literal_eval_value] = papers[literal_eval_value].apply(safe_literal_eval)  # jsonの読み込み
    return  papers

def to_dataframe(source):
    source = pd.DataFrame(source)
    return source

def is_valid_tiktoken(model_name, prompt):
    model, limit = tiktoken_dict[model_name]
    tokens = model.encode(prompt)
    if len(tokens) < limit:
        return True
    else:
        return False


def title_review_generate_prompt(abstracts, query_text, language):
    abstracts_text = "\n\n".join(abstracts)

    # Query:\n\n{query_text}を Academic Abstracts のあとに入れる？
    prompt = f"""Academic abstracts: \n\n {abstracts_text} 
                Instructions: These Abstracts are the most frequently referenced references in the literature and are assumed to provide background or theoretical perspectives.
                It is assumed that the smaller numbered papers are basic references that are referenced more often in the field, and that as the number increases, they become more closely related to a particular field (in this case, query).
                Note that the literature review will be structured so that the discussion begins with an understanding of the basic literature and then evolves to something closer to the query.
                Do three tasks:
                Task1: Write literature review of these abstract using all references.
                Task2: Discuss the latest developments in 5 years, specifying the publication year information in (published in ).
                Task3: What are the unmet medical needs in this relm in detail?
                Task4: What treatment methods are currently employed in detail?
                Do not use prior knowledge or your own assumptions.
                Make sure to cite results using [number] notation after the sentence.

                ## Summary ##

                ## Latest developments ##

                ## Unmet medical needs ##

                ## Treatments ##

                You must reply in English and write as long as possible.
                """

    japanese_prompt = f"""以下は学術論文のアブストラクトです。\n\n
                        {abstracts_text}\n
                        \n
                        これらのアブストラクトは{1}に関する研究分野の重要な知見をもたらしています。\n
                        これらに基づいて以下のタスクを実行してください。なお、事前知識や思い込みは使用しないで、水平思考で答えてください。\n
                        オーディエンスとしては、詳細な解析のための情報を必要とする製薬企業の開発担当者を想定してください。\n
                        \n
                        ## 定義\n
                        {query_text}について定義してください。情報があれば人口動態や罹患率についても述べてください。\n
                        \n
                        ## 要約\n
                        与えられたアブストラクトを包括的に要約してください。要約は、原文で提示されている重要なポイントや主要なアイデアをすべてカバーすると同時に、情報を簡潔で理解しやすい形式に凝縮する必要があります。不必要な情報や繰り返しを避けながら、主要なアイデアを裏付ける関連する詳細や例を含めるようにしてください。
                        要約の長さは、原文の長さと複雑さに対して適切であるべきで、重要な情報を省略することなく、明確で正確な概要を提供すること。\n
                        出力は20文出力して、行数を数えてください。\n
                        \n
                        ## 最近の発展\n
                        この研究分野の最近の発展について（Published in）に示される出版年を参考に述べてください。特に5年以内の発展を重視すること。\n
                        アブストラクトで提示されている重要なポイントや主要なアイデアをすべてカバーすると同時に、情報を簡潔で理解しやすい形式に凝縮する必要があります。
                        不必要な情報や繰り返しを避けながら、主要なアイデアを裏付ける関連する詳細や例を含めるようにしてください。文章の長さは、原文の長さと複雑さに対して適切であるべきで、重要な情報を省略することなく、明確で正確な概要を提供すること。\n
                        出力は20文出力して、行数を数えてください。\n
                        \n
                        ## 未解決の問題\n
                        これらのアブストラクトで指摘されている研究分野の未解決の問題をリストアップして、各項目について詳細に説明してください。\n
                        - \n
                        - \n
                        - \n
                        \n
                        ## 手法\n
                        これらのアブストラクトで指摘されているこの研究分野で用いられている手法や治療法とその効果をリストアップして、各項目について詳細に説明しtください。\n
                        - \n
                        - \n
                        - \n
                        \n
                        ## 薬剤\n
                        これらのアブストラクトで指摘されているこの研究分野で用いられている薬剤とその安全性・有効性をリストアップして、各項目について詳細に説明してください。\n
                        - \n
                        - \n
                        - \n
                        \n
                        回答にあたっては、[number]という形式で引用を明記してください。必ず日本語で回答してください。\n
                        では、深呼吸をして回答に取り組んでください。\n
                        """

    if language == '日本語':
        return japanese_prompt
    else:
        return prompt


#論文のレビュー部分
def title_review_papers(papers, query_text, model = 'gpt-4-32k', language="English"):
    abstracts = []
    linked_apas = []
    titles = []
    prompt = title_review_generate_prompt([], query_text, language)
    caption = f"{len(papers)} 件の論文がレビューに使用されました。"
    for i, (title, abstract, year, link) in enumerate(zip(papers["title"], papers["abstract"], papers['year'], papers['linked_APA']), 1):
        text = f"[{i}] (Published in {year}) {title}\n{abstract}"
        abstracts.append(text)
        linked_apas.append(f"[{i}] : {link}")
        titles.append(f"[{i}] : {title}\n{abstract}")

        prompt = title_review_generate_prompt(abstracts, query_text, language)
        if not is_valid_tiktoken(model, prompt):
            prompt = title_review_generate_prompt(abstracts[:-1], query_text, language)
            caption = f"{i - 1} / {len(papers)} 件の論文がレビューに使用されました。"
            break
    cluster_summary = get_azure_gpt_response(prompt, model)
    return cluster_summary, linked_apas, caption, titles


#論文のリバイズ部分
def summary_writer_generate_prompt(references, cluster_summary, draft, language, mode):


    references_text = "\n\n".join(references)

    if mode == "revise_and_add_citation":
        prompt = f"""Specific summary: \n\n 
                    {cluster_summary}\n\n 
    
                    References:\n\n
                    {references_text}\n\n
    
                    Instructions: Using the provided academic summaries, write a comprehensive long description about the given draft by synthesizing these summaries. You must write in English.
                    Make sure to cite results using [number] notation after the sentence. If the provided search results refer to multiple subjects with the same name, 
                    write separate answers for each subject.\n\n
    
                    Draft: {draft}\n\n
                    """

        japanese_prompt = f"""学術論文の要約：\n\n
                    {cluster_summary}\n\n
    
                    参考文献： \n\n
                    {references_text}\n\n
    
                    指示: 提供された学術論文の要約を総合して、与えられた草稿について包括的な長文の説明を記述しなさい。必ず日本語で記述しなさい。
                    参考文献の引用は、必ず文の後に[number]表記で行うこと。参考文献が同じ名前で複数の主題に言及している場合は、主題ごとに別々の解答を書きなさい。
    
                    草稿： {draft}
                    
                    """
    elif mode == "only_add_citation":
        prompt = f"""Academic abstracts: \n\n
                    {references_text}
                    \n\n
                    Instructions: Using the provided academic abstracts, do a task to each sentence in the given draft by minimizing content changes.
                    Task: Calculate the relevance between the sentence and the abstracts. Put [citation number] of top three most relevant abstract after the sentence. Then list up summaries of selected abstracts.
                    
                    ## Relevance Calculation:
                    
                    ## Draft with citations:
                    
                    ## Summaries of Selected Abstracts: 
                    
                    \n\n
                    Draft: {draft}"""

        japanese_prompt = f""": 学術抄録\n\n
                    {references_text}
                    \n\n
                    指示: 提供された学術抄録を使って、与えられた草稿の各文章に、内容の変更を最小限にしてタスクを実行しなさい。
                    タスク: 文章と抄録の関連性を計算する。最も関連性の高い抄録の上位3つの[citation number]を文の後ろに付ける。次に、選択した抄録の要約をリストアップする。
                    
                    ## 関連性の計算:
                    
                    ## 引用文献を含む草稿:
                    
                    ## 選択した抄録の要約: 
                    
                    \n\n
                    草稿： {draft}
                    """

    else:
        raise ValueError(f"Invalid mode {mode}")

    if language == '日本語':
        return japanese_prompt
    else:
        return prompt


def summery_writer_with_draft(cluster_summary, draft, references, model = 'gpt-4-32k',language="English", mode="only_add_citation"):
    prompt = summary_writer_generate_prompt([], cluster_summary, "", language, mode)
    if not is_valid_tiktoken( model, prompt):
        return "Cluster summary exceeds the AI characters limit. Please regenerate your cluster summary."

    prompt = summary_writer_generate_prompt([], cluster_summary, draft, language, mode)
    if not is_valid_tiktoken( model, prompt):
        return "Your draft exceeds the AI characters limit. Please shorten the length of your draft."


    caption = "All papers were used to generate the review. "
    for i in range(len(references)):
        prompt = summary_writer_generate_prompt(references[:i+1], cluster_summary, draft, language, mode)
        if not is_valid_tiktoken(model, prompt):
            prompt = summary_writer_generate_prompt(references[:i], cluster_summary, draft, language, mode)
            caption = f"{i -1 } / {len(references)} papers were used to generate the review. "
            break

    summary = get_azure_gpt_response(prompt, 'gpt-4-32k')
    return summary, caption


# DiGraphの初期化

def get_paper_graph(papers_df):
    # DiGraphの初期化
    G = nx.DiGraph()

    for k in tqdm.tqdm(range(len(papers_df))):  # 0 ~ len(papers_df)-1
        # ノードの追加
        if papers_df.loc[k, 'paperId'] is not None:
            G.add_node(papers_df.loc[k, 'paperId'], title=f"From:{papers_df.loc[k, 'title']}", year=papers_df.loc[k, 'year'])

        for reference in papers_df.loc[k, 'references']:
            if reference['paperId'] is not None:
                G.add_node(reference['paperId'], title=f"To:{reference['title']}", year=reference['year'])

        # エッジの追加
        for reference in papers_df.loc[k, 'references']:
            if reference['paperId'] is not None and papers_df.loc[k, 'paperId'] is not None:
                # XからYへ情報が流れる（YがXを引用する）X->Y
                # 出次数が多いものが被引用回数が高いもの
                G.add_edge(reference['paperId'], papers_df.loc[k, 'paperId'])
    return G

def extract_subgraph(G):
    # # 有向グラフの出次数が1より大きいノードだけを取得
    # large_degree = [node for node, out_degree in dict(G.out_degree()).items() if out_degree > 0]
    #
    # # サブグラフの作成
    # H_directed = G.subgraph(large_degree)
    #
    # # サブグラフを無向グラフに変換
    # H_undirected = H_directed.to_undirected()
    #
    # # 無向グラフの次数が0より大きいノードだけを取得
    # nodes_with_large_degree = [node for node, degree in dict(H_undirected.degree()).items() if degree > 0]
    #
    # # 新しいサブグラフの作成
    # H = H_undirected.subgraph(nodes_with_large_degree)

    H = G.to_undirected()
    return H


def plot_subgraph(H):
    # サブグラフの描画
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(H, seed=42)  # layoutの指定
    nx.draw(H, pos, with_labels=False, node_size=50)

    st.pyplot(plt)


#グラフ全体Hに対してPageRankを計算する
def get_cluster_papers(H, G, cluster_nodes):
    # クラスタ0に属するノードだけでサブグラフを作成
    H_cluster = H.subgraph(cluster_nodes)

    # PageRankの計算. 全体のページランクを使用する．
    pagerank = nx.pagerank(H, alpha=0.9)

    # 次数中心性の計算
    degree_centrality = nx.degree_centrality(H_cluster)

    # データフレームに変換
    df_centrality = pd.DataFrame(degree_centrality.items(), columns=['Node', 'DegreeCentrality'])

    # タイトルを持つ新しいカラムを作成
    df_centrality['TitleNode'] = df_centrality['Node'].map(nx.get_node_attributes(G, 'title'))
    df_centrality['Title'] = df_centrality['TitleNode'].str.replace('To:', '').str.replace('From:', '')

    # PageRankをデータフレームに追加
    df_centrality['PageRank'] = df_centrality['Node'].map(pagerank)

    # 次数中心性で降順ソート
    df_centrality = df_centrality.sort_values('PageRank', ascending=False)

    return df_centrality


def page_ranking_sort(H):
    # PageRankの計算
    pagerank = nx.pagerank(H, alpha=0.9)

    # 次数中心性の計算
    degree_centrality = nx.degree_centrality(H)

    # データフレームに変換
    df_centrality = pd.DataFrame(degree_centrality.items(), columns=['Node', 'DegreeCentrality'])

    # タイトルを持つ新しいカラムを作成
    df_centrality['Title'] = df_centrality['Node'].map(nx.get_node_attributes(H, 'title'))

    # 被引用回数を持つ新しいカラムを作成
    df_centrality['CitationCount'] = df_centrality['Node'].map(nx.get_node_attributes(H, 'citationCount'))

    # PageRankをデータフレームに追加F
    df_centrality['PageRank'] = df_centrality['Node'].map(pagerank)

    # 出版年をもつカラムを作成
    df_centrality['Year'] = df_centrality['Node'].map(nx.get_node_attributes(H, 'year'))

    # PageRankで降順ソート
    df_centrality = df_centrality.sort_values(['PageRank'], ascending=False)

    return df_centrality


#クラスタリングアルゴリズム
def community_clustering(H):
    # 最適なモジュラリティを持つコミュニティを見つける
    partition = community_louvain.best_partition(H, random_state=42)

    # クラスタのノード数をカウント
    cluster_counts = {}
    for cluster_id in partition.values():
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    print(f'cluster counts {cluster_counts}')

    clustering_result = {key: value for key, value in partition.items()}

    return cluster_counts, partition, clustering_result


def prepare_traces(G, partition):

    # ノードの位置を計算
    pos = nx.spring_layout(G, seed=42)  # シードの設定


    # ノードとエッジの描画
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), mode='lines', hoverinfo='none')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='none',
        # marker=dict(
        # showscale=True,
        # colorscale='YlGnBu',
        # color=[partition.get(node, 0) for node in G.nodes()],
        # size=10,
        # colorbar=dict(
        #     thickness=15,
        #     title='Community',
        #     xanchor='left',
        #     titleside='right'
        # )
        marker=dict(
        showscale=False,  # カラーバーを非表示
        colorscale='YlGnBu',
        color=[partition.get(node, 0) for node in G.nodes()],
        size=10)
    )

    return [edge_trace, node_trace]

def plot_subgraph_of_partition(H, partition):
    fig = go.Figure(data=prepare_traces(H, partition))

    # 図のサイズを10 x 10インチに設定
    fig.update_layout(width=10*72, height=10*72)  # 72 pixels per inch

    st.plotly_chart(fig)


def plot_cluster_i(H, cluster_id, partition):
    # クラスタ0番に属するノードを取得
    cluster_id_nodes = [node for node, cid in partition.items() if cid == cluster_id]
    if len(cluster_id_nodes) < 1:
        st.write("No nodes found for the given cluster ID.")
        return False
    # サブグラフの作成
    H_cluster_id = H.subgraph(cluster_id_nodes)

    # 描画の準備
    edge_trace_cluster_id, node_trace_cluster_id = prepare_traces(H_cluster_id, partition)  # 描画のための関数

    # 描画
    fig_cluster_id = go.Figure(data=[edge_trace_cluster_id, node_trace_cluster_id])
    fig_cluster_id.update_layout(
        width=10 * 72,
        height=10 * 72,
        # xaxis_title="X",  # X軸のラベルをここに設定
        # yaxis_title="Y",   # Y軸のラベルをここに設定
        margin = dict(l=0, r=0, b=0, t=0),  # 固定のマージン
        xaxis = dict(showticklabels=False),  # x軸の数字を非表示
        yaxis = dict(showticklabels=False)  # y軸の数字を非表示
    )

    st.plotly_chart(fig_cluster_id)
    return True


def extract_reference_indices(response):
    # 複数の数字を抽出するための正規表現を使用します。ハイフンでの範囲も考慮に入れます。
    numbers = re.findall(r'\[([\d,\s,-]+)\]', response)

    # 数字をコンマで分割し、整数型に変換してリストに格納します
    transformed_numbers = []
    for num_str in numbers:
        for num in num_str.split(","):
            num = num.strip()
            if '-' in num:  # ハイフンがある場合、その範囲のすべての数字を取得します。
                start, end = map(int, num.split('-'))
                transformed_numbers.extend(range(start, end + 2))
            else:
                transformed_numbers.append(int(num))

    # 1から始まるインデックスを0から始まるインデックスに変換
    transformed_numbers = [x - 1 for x in transformed_numbers]

    # 重複する数字を削除します
    transformed_numbers = list(set(transformed_numbers))

    # 数字をソートします
    transformed_numbers.sort()

    return transformed_numbers


def calc_tf_idf(df_centrality):
    cluster_keywords = defaultdict(list)

    # テキストを小文字化し、記号を取り除く
    df_centrality['CleanText'] = df_centrality['Title'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x.lower()))

    # TfidfVectorizerの初期化、ここでストップワードを指定
    vectorizer = TfidfVectorizer(stop_words='english')

    # クラスタごとにTF-IDFを計算
    for cluster in df_centrality['Cluster'].dropna().unique():
        texts = df_centrality[df_centrality['Cluster'] == cluster]['CleanText']
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # 平均TF-IDFスコアを計算
        avg_tfidf_scores = tfidf_matrix.mean(axis=0).tolist()[0]
        sorted_indices = np.argsort(avg_tfidf_scores)[::-1]

        for i in range(min(5, len(sorted_indices))):
            cluster_keywords[cluster].append(feature_names[sorted_indices[i]])

    return cluster_keywords


def bibtex_to_apa(bibtex_entry):
    apa_format = ""
    if not bibtex_entry or pd.isna(bibtex_entry):
        return ""
    bibtex_entry = bibtex_entry['bibtex']

    # Author
    author = re.search(r'author = {(.*?)}', bibtex_entry)
    if author:
        authors = author.group(1).split(' and ')
        formatted_authors = []
        for auth in authors:
            if " " in auth:  # Full name
                parts = auth.split(" ")
                last = parts[-1]
                initials = ". ".join([name[0] for name in parts[:-1]])
                formatted_authors.append(f"{last}, {initials}.")
            else:  # Initials
                formatted_authors.append(auth)

        if len(formatted_authors) == 1:
            apa_format += formatted_authors[0]
        elif len(formatted_authors) == 2:
            apa_format += f"{formatted_authors[0]} & {formatted_authors[1]}"
        else:
            apa_format += ", ".join(formatted_authors[:-1]) + f" & {formatted_authors[-1]}"

    # Year
    year = re.search(r'year = {(.*?)}', bibtex_entry)
    if year:
        apa_format += f" ({year.group(1)}). "

    # Title
    title = re.search(r' title = {(.*?)}', bibtex_entry)
    if title:
        apa_format += f"{title.group(1)}. "

    # Journal
    journal = re.search(r'journal = {(.*?)}', bibtex_entry)
    if journal:
        apa_format += f"{journal.group(1)}"

    # Volume
    volume = re.search(r'volume = {(.*?)}', bibtex_entry)
    if volume:
        apa_format += f", {volume.group(1)}"

    # Pages
    pages = re.search(r'pages = {(.*?)}', bibtex_entry)
    if pages:
        cleaned_pages = pages.group(1).replace('\n', '').strip()
        apa_format += f", pp. {cleaned_pages}."
    else:
        apa_format += '.'

    return apa_format

def set_paper_information(papers):
    papers['APA'] = papers['citationStyles'].apply(bibtex_to_apa)
    papers['linked_APA'] = papers.apply(lambda
                                            row: f'<a href="https://www.semanticscholar.org/paper/{row["paperId"]}" target="_blank">{row["APA"]}</a>',
                                        axis=1)
    return papers


def save_values():
    print(st.session_state)
    st.experimental_set_query_params(**st.session_state)
    params = st.experimental_get_query_params()
    print(params)


if __name__=='__main__':
    # response = "近年、社会ロボットやAIとのインタラクションが人間の感情調整に効果的であることが明らかとなりました。特に、これらの技術が子供たちの感情調整をサポートする能力が注目されています。" \
    #            "一部の研究では、マルチ重み付けマルコフ決定プロセス感情調整（MDPER）ロボットが作成され、その結果、人間とロボット間の感情調整が可能となっています[1]。" \
    #            "加えて、子供たちが社会ロボットとのストーリーテリング活動で創造性を発揮する際に感情調整技術が有効に使われています[2]。" \
    #             "このような発展により、「感情の視覚表現や温度の表現」を通じてロボットが感情を伝達し、子供の感情調節を支援するという新しい可能性が生まれてきました[13,20,16,7]。" \
    #            "具体的には、社会ロボットの感情的なアイジェスチャーを設計するフレームワークが研究されており[20]、このフレームワークを使用すれば、ロボットは視覚的表現を通じて子供たちに自らの感情状態を伝達できるようになるでしょう。" \
    #            "子供たちが自身の感情をより効果的に制御できるようになったと報告されています[10-13]。" \
    #            "また、自閉スペクトラム障害（ASD）の子供たちも、社会感情スキルとSTEMの学習を同時に進めるために、AIとのインタラクションを通じた学習が行われており、有用性が確認されています[4]。" \
    #            "しかし、それらの技術や方法がまだ発展段階にあることを忘れてはなりません。より効果的な感情調整手段、具体的な介入方法、効率的な学習手法を見つけて、これらのツールを生かすためには、引き続き研究が必要となります。"
    #
    # print(extract_reference_indices(response))
    set_button = st.button("SET")
    if set_button:
        st.session_state['a'] = 0
        st.session_state['b'] = {"1": 2, (4,): 3}
    save_values()
    refresh_button = st.button("REFRESH")
    if refresh_button:
        st.experimental_rerun()