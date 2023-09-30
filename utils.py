import requests
import os
from time import sleep
import tiktoken


current_dir = os.path.abspath(os.path.dirname(__file__))
data_folder = os.path.join(current_dir, 'database')
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import community as community_louvain # python-louvain packageをインストールする必要があるわ
import plotly.graph_objects as go
import networkx as nx
import openai
import streamlit as st
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
DEEPL_API_KEY = st.secrets['DEEPL_API_KEY']
SEMANTICSCHOLAR_API_KEY = st.secrets['SEMANTICSCHOLAR_API_KEY']
    
openai.api_key = OPENAI_API_KEY

def tiktoken_setup(offset = 8):
    gpt_35_tiktoken = tiktoken.encoding_for_model("gpt-3.5-turbo")
    gpt_35_16k_tiktoken = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
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

def get_gpt_response(system_input, model = "gpt-4"):
    print(f'prompt\n {system_input}')
    response = openai.ChatCompletion.create(
        model= model,
        messages=[
          {"role": "system", "content": system_input},
        ],
    )
    print(response.choices[0]["message"]["content"].strip())
    return response.choices[0]["message"]["content"].strip()

def get_gpt_response2(system_input, user_input, model = "gpt-3.5-turbo-16k"):
    response = openai.ChatCompletion.create(
        model= model,
        messages=[
          {"role": "system", "content": system_input},
            {"role": "user", "content" : user_input}
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
def _get_papers(query, offset, limit, fields):
    base_url = "http://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "offset": offset, "limit": limit, "fields": fields}
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

def get_papers(query_text, offset = 0, limit = 100, total_limit = 1000):
  papers = []
  fields = "paperId,title,abstract,year,authors,journal,citationCount,referenceCount,references,embedding,references.title,references.abstract,references.year,references.authors,references.citationCount,references.referenceCount"

  # 最初の結果セットを取得
  # _get_papersはエラーが発生した場合 None を返す
  result = _get_papers(query_text, offset, limit, fields=fields)
  if not result:
    return [], 0

  papers.extend(result['data'])
  print(f"Total results is {result['total']}.")
  print(f"{min(round(limit / min(total_limit, result['total']) * 100, 1), 100)}%")

  # 1000件未満なら全件取得
  while len(papers) < min(total_limit, result['total']):
      offset += limit
      result = _get_papers(query_text, offset, limit, fields=fields)
      if not result:
          if len(papers) > 0:
              break
          else:
            return [], 0
      print(f"{min(round(min(total_limit, offset + limit) / min(total_limit, result['total']) * 100, 1), 100)}%")
      papers.extend(result['data'])
  return papers, result['total']

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
    fields = "paperId,title,abstract,year,authors,journal,citationCount,referenceCount,references,embedding,references.title,references.abstract,references.year,references.authors,references.citationCount,references.referenceCount"

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

def get_keywords(user_input):
    system_input = "You will be provided with a block of text, and your task is to extract a list of keywords from it. Please output only keywords separated by commas."
    try:
        gpt_response = get_gpt_response2(system_input, user_input, 'gpt-3.5-turbo-16k')
        print(f'gpt response is {gpt_response}')
        keywords = gpt_response.split(',')
        return keywords
    except Exception as e:
        print(f'Error happened as {e}')
        sleep(5)

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

def save_papers_dataframe(df, query_text):
    file_name = safe_filename(query_text)
    df.to_csv(os.path.join(data_folder, f'{file_name}.csv'), encoding='utf-8')


def load_papers_dataframe(query_text):
    file_name = safe_filename(query_text)
    papers = pd.read_csv(os.path.join(data_folder, f'{file_name}.csv'))
    papers.dropna(subset=['embedding', 'title', 'abstract'], inplace=True)
    papers.reset_index(drop=True, inplace=True)

    papers['references'] = papers['references'].apply(ast.literal_eval)  # jsonの読み込み
    papers['authors'] = papers['authors'].apply(ast.literal_eval)
    papers['embedding'] = papers['embedding'].apply(ast.literal_eval)

    return  papers

def to_dataframe(source, drop_list = [ 'title', 'abstract']):
    source = pd.DataFrame(source)
    if len(drop_list) > 0:
        source.dropna(subset=drop_list, inplace=True)
        source.reset_index(drop=True, inplace=True)
    return source

# def topk_review_papers(papers_df, query_text, topk=20):
#     _papers_df = papers_df.head(topk)
#     texts = []
#     for i, (title, abstract) in enumerate(zip(_papers_df["title"], _papers_df["abstract"]), 1):
#         text = f"[{i}] {title} - {abstract}"
#         texts.append(text)
#
#     full_text = "\n\n".join(texts)
#
#     # アブストラクトをもとに要約を生成する指示を出す
#     prompt = f"""Academic abstracts: \n\n {full_text} \n\n Instructions: Using the provided academic abstracts, write a comprehensive description about the given query by synthesizing these OBJECTIVE. Make sure to cite results using [number] notation after the sentence. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.\n\nQuery: {query_text}"""
#     gpt_response = get_gpt_response(prompt, 'gpt-3.5-turbo-16k')
#     return gpt_response

def is_valid_tiktoken(model_name, prompt):
    model, limit = tiktoken_dict[model_name]
    tokens = model.encode(prompt)
    if len(tokens) < limit:
        return True
    else:
        return False

#tiktoken版
def topk_review_generate_prompt(abstracts, query_text):
    abstracts_text = "\n\n".join(abstracts)
    prompt = f"""Academic abstracts: \n\n {abstracts_text} \n\n Instructions: Using the provided academic abstracts, write a comprehensive description about the given query by synthesizing these OBJECTIVE. Make sure to cite results using [number] notation after the sentence. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.\n\nQuery: {query_text}"""
    return prompt

def topk_review_papers(papers_df, query_text, model='gpt3.5-turbo-16k', topk=20):
    abstracts = []
    _papers_df = papers_df.head(topk)
    caption = "All papers were used to generate the review. "

    for i, (title, abstract) in enumerate(zip(_papers_df["title"], _papers_df["abstract"]), 1):
        text = f"[{i}] {title} - {abstract}"
        abstracts.append(text)
        prompt = topk_review_generate_prompt(abstracts, query_text)

        if not is_valid_tiktoken(model, prompt):
            prompt = topk_review_generate_prompt(abstracts[:-1], query_text)
            caption = f"{i -1 } / {topk} papers were used to generate the review. "
            break
    gpt_response = get_gpt_response(prompt, model)
    return gpt_response, caption



def get_cluster_papers(H, G, cluster_nodes):
    # クラスタ0に属するノードだけでサブグラフを作成
    H_cluster = H.subgraph(cluster_nodes)

    # 次数中心性を計算
    degree_centrality = nx.degree_centrality(H_cluster)

    # データフレームに変換
    df_centrality = pd.DataFrame(degree_centrality.items(), columns=['Node', 'DegreeCentrality'])

    # タイトルを持つ新しいカラムを作成
    df_centrality['TitleNode'] = df_centrality['Node'].map(nx.get_node_attributes(G, 'title'))
    df_centrality['Title'] = df_centrality['TitleNode'].str.replace('To:', '').str.replace('From:', '')

    # 次数中心性で降順ソート
    df_centrality = df_centrality.sort_values('DegreeCentrality', ascending=False)
    return df_centrality

# def title_review_papers(papers, query_text, language="English"):
#     texts = []
#     titles = []
#     for i, (title, abstract) in enumerate(zip(papers["title"], papers["abstract"]), 1):
#         text = f"[{i}] {title}\n{abstract}"
#         texts.append(text)
#         titles.append(f'[{i}] {title}')
#
#     full_cluster_text = "\n\n".join(texts)
#
#     if language == '日本語':
#         language_prompt = "Output in Japanese.\n\n"
#     else:
#         language_prompt = ""
#
#     prompt = f"""Academic abstracts: \n\n {full_cluster_text}
#                 Instructions: These Abstracts are the most frequently referenced references in the literature and are assumed to provide background or theoretical perspectives.
#                 It is assumed that the smaller numbered papers are basic references that are referenced more often in the field, and that as the number increases, they become more closely related to a particular field (in this case, query).
#                 Note that the literature review will be structured so that the discussion begins with an understanding of the basic literature and then evolves to something closer to the query.
#                 Do three tasks:
#                 Task1: Write literature review of these abstract.
#                 Task2: What are the unsolved problems in this relm?
#                 Task3: Point out implication to {query_text}.
#                 Do not use prior knowledge or your own assumptions.
#                 Make sure to cite results using [number] notation after the sentence.
#                 {language_prompt}
#
#                 ## Summary ##
#
#                 ## Unsolved problems ##
#
#                 ## Implication ##
#
#                 """
#
#
#     cluster_summary = get_gpt_response(prompt, 'gpt-3.5-turbo-16k')
#     return cluster_summary, titles


def title_review_generate_prompt(abstracts, query_text, language):
    if language == '日本語':
        language_prompt = "Output in Japanese.\n\n"
    else:
        language_prompt = ""

    abstracts_text = "\n\n".join(abstracts)

    prompt = f"""Academic abstracts: \n\n {abstracts_text} 
                Instructions: These Abstracts are the most frequently referenced references in the literature and are assumed to provide background or theoretical perspectives. 
                It is assumed that the smaller numbered papers are basic references that are referenced more often in the field, and that as the number increases, they become more closely related to a particular field (in this case, query). 
                Note that the literature review will be structured so that the discussion begins with an understanding of the basic literature and then evolves to something closer to the query.
                Do three tasks:
                Task1: Write literature review of these abstract. 
                Task2: What are the unsolved problems in this relm?
                Task3: Point out implication to {query_text}.
                Do not use prior knowledge or your own assumptions.
                Make sure to cite results using [number] notation after the sentence.


                ## Summary ##

                ## Unsolved problems ##

                ## Implication ##
                
                {language_prompt}
                """
    return prompt

def title_review_papers(papers, query_text, model = 'gpt-3.5-turbo-16k', language="English"):
    abstracts = []
    titles = []
    prompt = title_review_generate_prompt([], query_text, language)
    caption = f"All papers were used to generate the review. "
    for i, (title, abstract) in enumerate(zip(papers["title"], papers["abstract"]), 1):
        text = f"[{i}] {title}\n{abstract}"
        abstracts.append(text)
        titles.append(f'[{i}] {title}')

        prompt = title_review_generate_prompt(abstracts, query_text, language)
        if not is_valid_tiktoken(model, prompt):
            prompt = topk_review_generate_prompt(abstracts[:-1], query_text)
            caption = f"{i - 1} / {len(papers)} papers were used to generate the review. "
            break
    cluster_summary = get_gpt_response(prompt, model)
    return cluster_summary, titles, caption


def summary_writer_generate_prompt(references, cluster_summary, draft, language):
    if language == '日本語':
        language_prompt = "Output in Japanese.\n\n"
    else:
        language_prompt = ""

    references_text = "\n\n".join(references)

    prompt = f"""Specific summary: \n\n 
                {cluster_summary}\n\n 

                References:\n\n
                {references_text}\n\n

                Instructions: Using the provided academic summaries, write a comprehensive long description about the given draft by synthesizing these summaries. 
                Make sure to cite results using [number] notation after the sentence. If the provided search results refer to multiple subjects with the same name, 
                write separate answers for each subject.\n\n

                Draft: {draft}\n\n
                
                {language_prompt}
                """
    return prompt
def summery_writer_with_draft(cluster_summary, draft, references, model = 'gpt-3.5-turbo-16k',language="English"):
    prompt = summary_writer_generate_prompt([], cluster_summary, "", language)
    if not is_valid_tiktoken( model, prompt):
        return "Cluster summary exceeds the AI characters limit. Please regenerate your cluster summary."

    prompt = summary_writer_generate_prompt([], cluster_summary, draft, language)
    if not is_valid_tiktoken( model, prompt):
        return "Your draft exceeds the AI characters limit. Please shorten the length of your draft."


    caption = "All papers were used to generate the review. "
    for i in range(len(references)):
        prompt = summary_writer_generate_prompt(references[:i+1], cluster_summary, draft, language)
        if not is_valid_tiktoken(model, prompt):
            prompt = summary_writer_generate_prompt(references[:i], cluster_summary, draft, language)
            caption = f"{i -1 } / {len(references)} papers were used to generate the review. "
            break

    summary = get_gpt_response(prompt, 'gpt-3.5-turbo-16k')
    return summary, caption

# DiGraphの初期化

def get_paper_graph(papers_df):
    # DiGraphの初期化
    G = nx.DiGraph()

    for k in tqdm.tqdm(range(len(papers_df))):  # 0 ~ len(papers_df)-1
        # ノードの追加
        if papers_df.loc[k, 'paperId'] is not None:
            G.add_node(papers_df.loc[k, 'paperId'], title=f"From:{papers_df.loc[k, 'title']}")

        for reference in papers_df.loc[k, 'references']:
            if reference['paperId'] is not None:
                G.add_node(reference['paperId'], title=f"To:{reference['title']}")

        # エッジの追加
        for reference in papers_df.loc[k, 'references']:
            if reference['paperId'] is not None and papers_df.loc[k, 'paperId'] is not None:
                # XからYへ情報が流れる（YがXを引用する）X->Y
                # 出次数が多いものが被引用回数が高いもの
                G.add_edge(reference['paperId'], papers_df.loc[k, 'paperId'])
    return G

def extract_subgraph(G):
    # 有向グラフの出次数が1より大きいノードだけを取得
    large_degree = [node for node, out_degree in dict(G.out_degree()).items() if out_degree > 0]

    # サブグラフの作成
    H_directed = G.subgraph(large_degree)

    # サブグラフを無向グラフに変換
    H_undirected = H_directed.to_undirected()

    # 無向グラフの次数が0より大きいノードだけを取得
    nodes_with_large_degree = [node for node, degree in dict(H_undirected.degree()).items() if degree > 0]

    # 新しいサブグラフの作成
    H = H_undirected.subgraph(nodes_with_large_degree)

    return H


def plot_subgraph(H):
    # サブグラフの描画
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(H, seed=42)  # layoutの指定
    nx.draw(H, pos, with_labels=False, node_size=50)

    # タイトルのラベルを追加
    # for p in pos:  # ノードの位置情報を利用
    #     pos[p][1] += 0.07
    # nx.draw_networkx_labels(H, pos, labels=nx.get_node_attributes(H, 'title'), font_size=8)

    st.pyplot(plt)

def community_clustering(H):
    # 最適なモジュラリティを持つコミュニティを見つける
    partition = community_louvain.best_partition(H, random_state=42)

    # クラスタのノード数をカウント
    cluster_counts = {}
    for cluster_id in partition.values():
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    print(f'cluster counts {cluster_counts}')

    # データフレームの作成
    cluster_df = pd.DataFrame(list(cluster_counts.items()), columns=['clusterNumber', 'numberOfNodes'])


    # クラスタリング係数、平均経路長、密度を計算
    clustering_coefficients = []
    average_path_lengths = []
    densities = []

    for community in set(partition.values()):
        subgraph = H.subgraph([node for node in partition if partition[node] == community])
        clustering_coefficient = nx.average_clustering(subgraph)
        clustering_coefficients.append(clustering_coefficient)

        if nx.is_connected(subgraph):
            average_path_length = nx.average_shortest_path_length(subgraph)
        else:
            average_path_length = None
        average_path_lengths.append(average_path_length)

        density = nx.density(subgraph)
        densities.append(density)

    # 既存のデータフレームにこれらの値を追加
    cluster_df['clusteringCoefficient'] = clustering_coefficients
    cluster_df['averagePathLength'] = average_path_lengths
    cluster_df['density'] = densities
    return cluster_counts, cluster_df, partition


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

    st.write(f'<h4> Number of papers used in the plot: {len(cluster_id_nodes)} </h4>', unsafe_allow_html=True)

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
    # 数字を抽出します
    numbers = re.findall(r'\[(\d+)\]', response)
    # 数字から1を引き、整数型に変換してリストに格納します
    transformed_numbers = list(set([int(num) - 1 for num in numbers]))
    # 数字をソートします
    transformed_numbers.sort()
    return transformed_numbers