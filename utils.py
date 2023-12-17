import requests
import os
from time import sleep
import tiktoken
current_dir = os.path.abspath(os.path.dirname(__file__))
debug_data_folder = os.path.join(current_dir, 'debug_database')
data_folder = os.path.join(current_dir, 'database')
import re
import ast
import pandas as pd
import random
import matplotlib.pyplot as plt
from functools import partial
from scipy.special import comb # 組み合わせ数を計算する関数
import tqdm
import plotly.graph_objects as go
import networkx as nx
import openai
from openai import OpenAI
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import community as community_louvain # python-louvain packageをインストールする必要があるわ
import json
import itertools
import fitz  # PyMuPDF

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
DEEPL_API_KEY = st.secrets['DEEPL_API_KEY']
SEMANTICSCHOLAR_API_KEY = st.secrets['SEMANTICSCHOLAR_API_KEY']
AZURE_OPENAI_API_KEY = st.secrets['AZURE_OPENAI_KEY']
AZURE_OPENAI_ENDPOINT = st.secrets['AZURE_OPENAI_ENDPOINT']

# OPEN AI の API を設定する場合
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# AZURE OPEN AI の API を設定する場合
# openai.api_key = AZURE_OPENAI_API_KEY
# openai.api_base = AZURE_OPENAI_ENDPOINT
# openai.api_type = 'azure'
# openai.api_version = '2023-05-15'


def tiktoken_setup(offset = 8):
    gpt_35_tiktoken = tiktoken.encoding_for_model("gpt-3.5-turbo")
    gpt_35_16k_tiktoken = tiktoken.encoding_for_model("gpt-4-32k")
    gpt_4_tiktoken = tiktoken.encoding_for_model("gpt-4")
    gpt_4_32k_tiktoken = tiktoken.encoding_for_model("gpt-4-32k")
    gpt_4_turbo_tiktoken = tiktoken.encoding_for_model("gpt-4-1106-preview")

    tiktoken_dict = {
        "gpt-3.5-turbo": (gpt_35_tiktoken, 4097 - offset),
        "gpt-3.5-turbo-16k": (gpt_35_16k_tiktoken, 16385 - offset),
        "gpt-3.5-turbo-1106": (gpt_35_tiktoken, 16385 - offset),
        "gpt-4": (gpt_4_tiktoken, 8192 - offset),
        "gpt-4-32k": (gpt_4_32k_tiktoken, 32768 - offset),
        "gpt-4-1106-preview": (gpt_4_turbo_tiktoken, 128000 - offset)
    }
    return tiktoken_dict

tiktoken_dict = tiktoken_setup()


def get_semantic_scholar_fields(expansion=True):
    fields = ["paperId",
              "title",
              "abstract",
              "year",
              "authors",
              "journal",
              "isOpenAccess",
              "openAccessPdf",
              "citationCount",
              "citationStyles",
              "referenceCount",
              ]

    if expansion:
        expand_fields = fields + ["references"] + [f"references.{field}" for field in fields]
        expand_fields = expand_fields + ["citations"] + [f"citations.{field}" for field in fields]
        return ",".join(expand_fields)
    else:
        return ','.join(fields)

def get_azure_gpt_response(system_input, model_name='gpt-4-32k'):
    response = client.chat.completions.create(
        model= model_name,
        messages=[
          {"role": "system", "content": system_input},
        ],
    )
    return response.choices[0]["message"]["content"].strip()

def get_azure_gpt_response_stream(system_input, model_name='gpt-4-32k'):
    response = client.chat.completions.create(
        model= model_name,
        messages=[
          {"role": "system", "content": system_input},
        ],
        stream = True
    )
    return response

def get_azure_gpt_message_stream(messages, model_name='gpt-4-32k'):
    response = client.chat.completions.create(
        model= model_name,
        messages=
          messages
        ,
        stream = True
    )
    return response

def get_azure_gpt_response_json(system_input, model_name='gpt-4'):
    if 'gpt-3.5' in model_name:
        model = 'gpt-3.5-turbo-1106'
    elif 'gpt-4' in model_name:
        model = 'gpt-4-1106-preview'
    else:
        raise ValueError(f"model name {model_name}")
    response = client.chat.completions.create(
        model= model,
        messages=[
          {"role": "system", "content": system_input},
        ],
        response_format={"type" : 'json_object'}
    )
    # print(response)
    return json.loads(response.choices[0].message.content)

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
def _get_papers(streamlit_empty, query, year, offset, limit, fields):
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
                streamlit_empty.write(f"タイムアウトしました。再取得中です...")
                print(f"Request timed out, retrying... {retries} attempts left")
                retries -= 1
                sleep(1)  # タイムアウトした場合、少し待つ
            else:
                streamlit_empty.write(f"論文の取得に失敗しました。リクエストを分割して取得します。")
                print(f"Request failed with status {response.status_code}")
                print(f"Request error message {response.reason}")
                print(f"Request content {response.content}")
                return None
        except Exception as e:
            print(f"ERROR HAPPENED AS {e}")
            return None

    print("All retries failed")
    return None

def get_papers(streamlit_empty, query_text, year, offset = 0, limit = 100, total_limit = 1000):
  papers = []
  fields = get_semantic_scholar_fields()

  # 最初の結果セットを取得
  result = _get_papers(streamlit_empty, query_text, year, offset, limit, fields=fields)
  if not result:
      return get_papers(streamlit_empty, query_text, year, offset, limit = int(limit / 5), total_limit = total_limit)
  if not result or result['total'] == 0:
    return [], 0


  papers.extend(result['data'])
  print(f"Total results is {result['total']}.")
  streamlit_empty.write(f"{min(round(limit / min(total_limit, result['total']) * 100, 1), 100)}%の論文を取得しました。")
  print(f"{min(round(limit / min(total_limit, result['total']) * 100, 1), 100)}%")

  # 1000件未満なら全件取得
  while len(papers) < min(total_limit, result['total']):
      offset += limit
      result = _get_papers(streamlit_empty, query_text, year, offset, limit, fields=fields)
      if not result:
          return get_papers(streamlit_empty, query_text, year, offset, limit=int(limit / 5), total_limit=total_limit)
      if not result:
          if len(papers) > 0:
              break
          else:
             return [], 0
      print(f"{min(round(min(total_limit, offset + limit) / min(total_limit, result['total']) * 100, 1), 100)}%")
      streamlit_empty.write(f"{min(round(min(total_limit, offset + limit) / min(total_limit, result['total']) * 100, 1), 100)}%の論文を取得しました。")
      papers.extend(result['data'])
  streamlit_empty.empty()
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

        citations = row['citations']
        if citations:
            for citation in citations:
                cite_dict = {"paperId" : row['paperId']}
                cite_dict.update(citation)
                reference_df_list.append(cite_dict)

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


def get_papers_from_ids(paper_ids, offset=0, limit=20):
    total_results = []
    total_dict = {}
    fields = get_semantic_scholar_fields(expansion=False)

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
    safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', filename)
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
    df.to_csv(os.path.join(debug_data_folder, f'{file_name}.csv'), encoding='utf-8')


def load_papers_dataframe(query_text, literal_evals = ['references', 'authors', 'citationStyles']):
    encoded_query_text_without_extension = encode_to_filename(query_text)
    file_name = safe_filename(encoded_query_text_without_extension)
    papers = pd.read_csv(os.path.join(debug_data_folder, f'{file_name}.csv'))
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

def title_long_review_generate_prompt(abstracts, query_text, language):
    abstracts_text = "\n\n".join(abstracts)

    if language == 'English':
        long_templates = []

        templates_header = f"""
            You are a researcher working on a literature review. The following are abstracts of academic papers to be reviewed.\n\n
            {abstracts_text}\n\n
            Based on these, please perform the following tasks to create a literature review on {query_text}. Do not use prior knowledge or assumptions, and answer with lateral thinking.\n
            Describe as instructed in the items marked with '##'.
            \n
        """

        long_templates.append(templates_header + f"""
            Please define {query_text} within 1000 characters. If available, also mention demographics and prevalence.\n
            In your answer, please cite using the format [number]. If the abstract mentions OpenAccess, write it as [number OpenAccess]. Please answer in Japanese.\n
            Try to make your response as lengthy as possible. Now, take a deep breath and start answering.\n
            \n
            ## Definition\n
            """)

        long_templates.append(templates_header + """
            Organize the content of the given abstracts and describe them comprehensively within 1000 characters. Cover all important points and main ideas presented in the original text, while condensing the information into a concise and understandable format. Include relevant details and examples that support the main ideas, avoiding unnecessary information and repetition. The length should be sufficiently long, providing a clear and accurate summary without omitting important information.\n
            In your answer, please cite using the format [number]. If the abstract mentions OpenAccess, write it as [number OpenAccess]. Please answer in Japanese.\n
            Try to make your response as lengthy as possible. Now, take a deep breath and start answering.\n
            \n
            ## Summary\n
            """)

        long_templates.append(templates_header + """
            Discuss recent developments in this research field within 1000 characters, considering the publication year indicated in (Published in). Place particular emphasis on developments within the last five years.\n
            Cover all important points and main ideas presented in the abstracts, while condensing the information into a concise and understandable format. Include relevant details and examples that support the main ideas, avoiding unnecessary information and repetition. The length should be sufficiently long, providing a clear and accurate summary without omitting important information.\n
            In your answer, please cite using the format [number]. If the abstract mentions OpenAccess, write it as [number OpenAccess]. Please answer in Japanese.\n
            Try to make your response as lengthy as possible. Now, take a deep breath and start answering.\n
            \n
            ## Recent Developments\n
            """)

        long_templates.append(templates_header + """
            List and explain in detail the unresolved issues in the research field pointed out in these abstracts, within 1000 characters for each item.\n
            Information on unresolved issues should be as comprehensive as possible. If you cannot reflect all information from the abstracts, state 'References to other unresolved issues are also found in other abstracts ([number][number][number]...)' and indicate the [number] of those abstracts.\n
            -
            -
            -
            In your answer, please cite using the format [number]. If the abstract mentions OpenAccess, write it as [number OpenAccess]. Please answer in Japanese.\n
            Try to make your response as lengthy as possible. Now, take a deep breath and start answering.\n
            \n
            ## Unresolved Issues\n
            """)

        long_templates.append(templates_header + """
            List and explain in detail the methods and treatments used in this research field and their effects, as pointed out in these abstracts, within 1000 characters for each item.\n
            Information on methods and treatments should be as comprehensive as possible. If you cannot reflect all information from the abstracts, state 'References to other methods/treatments are also found in other abstracts ([number][number][number]...)' and indicate the [number] of those abstracts.\n
            -
            -
            -
            In your answer, please cite using the format [number]. If the abstract mentions OpenAccess, write it as [number OpenAccess]. Please answer in Japanese.\n
            Try to make your response as lengthy as possible. Now, take a deep breath and start answering.\n
            \n
            ## Methods\n
            """)

        long_templates.append(templates_header + """
            List and explain in detail the drugs used in this research field and their safety and efficacy, as pointed out in these abstracts, within 1000 characters for each item.\n
            Information on drugs should be as comprehensive as possible. If you cannot reflect all information from the abstracts, state 'References to other drugs are also found in other abstracts ([number][number][number]...)' and indicate the [number] of those abstracts.\n
            -
            -
            -
            In your answer, please cite using the format [number]. If the abstract mentions OpenAccess, write it as [number OpenAccess]. Please answer in Japanese.\n
            Try to make your response as lengthy as possible. Now, take a deep breath and start answering.\n
            \n
            ## Drugs\n
            """)
        return long_templates

    if language == '日本語':
        long_templates = []

        templates_header = f"""
        あなたは文献レビューに取り組む研究者です。以下は文献レビューの対象となる学術論文のアブストラクトです。\n\n
        {abstracts_text}
        \n\n
        これらに基づいて、{query_text}についての文献レビューを作成するための以下のタスクを実行してください。なお、事前知識や思い込みは使用しないで、水平思考で答えてください。\n
        オーディエンスとしては、詳細な解析のための情報を必要とする研究者を想定してください。\n
        以下の「## 」で記された項目について指示通りに述べなさい。
        \n
        """

        long_templates.append(templates_header + f"""
        {query_text}について1000字以内で定義してください。情報があれば人口動態や罹患率についても述べてください。\n
        回答にあたっては、[number]という形式で引用を明記してください。日本語で回答してください。\n
        文章の分量はできるだけ多くしてください。では、深呼吸をして回答に取り組んでください。\n
        \n
        """)

        long_templates.append(templates_header + """
        与えられたアブストラクトの内容を整理して、1000字以内で包括的に記述してください。原文で提示されている重要なポイントや主要なアイデアをすべてカバーすると同時に、情報を簡潔で理解しやすい形式に凝縮する必要があります。不必要な情報や繰り返しを避けながら、主要なアイデアを裏付ける関連する詳細や例を含めるようにしてください。長さは十分に長くであるべきで、重要な情報を省略することなく、明確で正確な概要を提供すること。\n
        回答にあたっては、[number]という形式で引用を明記してください。日本語で回答してください。\n
        文章の分量はできるだけ多くしてください。では、深呼吸をして回答に取り組んでください。\n
        \n
        """)

        long_templates.append(templates_header + """
        この研究分野の最近の発展について（Published in）に示される出版年を参考に1000字以内で述べてください。特に5年以内の発展を重視すること。\n
        アブストラクトで提示されている重要なポイントや主要なアイデアをすべてカバーすると同時に、情報を簡潔で理解しやすい形式に凝縮する必要があります。不必要な情報や繰り返しを避けながら、主要なアイデアを裏付ける関連する詳細や例を含めるようにしてください。文章の長さは、十分に長くあるべきで、重要な情報を省略することなく、明確で正確な概要を提供すること。\n
        回答にあたっては、[number]という形式で引用を明記してください。日本語で回答してください。\n
        文章の分量はできるだけ多くしてください。では、深呼吸をして回答に取り組んでください。\n
        \n
        """)

        long_templates.append(templates_header + """
        これらのアブストラクトで指摘されている研究分野の未解決の問題をリストアップして、各項目について1000字以内で詳細に説明してください。\n
        未解決の問題に関する情報は包括的であればあるほど望ましいです。もしアブストラクトの情報をすべて反映できない場合には「他の未解決の問題についての言及も他のアブストラクトの中にあります（[number][number][number]...）」とそのアブストラクトの[number]を示して述べてください。\n
        -
        -
        -
        回答にあたっては、[number]という形式で引用を明記してください。日本語で回答してください。\n
        文章の分量はできるだけ多くしてください。では、深呼吸をして回答に取り組んでください。\n
        \n
        """)

        long_templates.append(templates_header + """
        これらのアブストラクトで指摘されているこの研究分野で用いられている手法や治療法とその効果をリストアップして、各項目について1000字以内で詳細に説明しtください。\n
        手法や治療法の情報は包括的であればあるほど望ましいです。もしアブストラクトの情報をすべて反映できない場合には「他の手法・治療法についての言及も他のアブストラクトの中にあります（[number][number][number]...）」とそのアブストラクトの[number]を示して述べてください。\n
        -
        -
        -
        回答にあたっては、[number]という形式で引用を明記してください。日本語で回答してください。\n
        文章の分量はできるだけ多くしてください。では、深呼吸をして回答に取り組んでください。\n
        \n
        """)

        long_templates.append(templates_header + """
        これらのアブストラクトで指摘されているこの研究分野で用いられている薬剤とその安全性・有効性をリストアップして、各項目について1000字以内で詳細に説明してください。\n
        薬剤の情報は包括的であればあるほど望ましいです。もしアブストラクトの情報をすべて反映できない場合には「他の薬剤についての言及も他のアブストラクトの中にあります（[number][number][number]...）」とそのアブストラクトの[number]を示して述べてください。\n
        -
        -
        -
        回答にあたっては、[number]という形式で引用を明記してください。日本語で回答してください。\n
        文章の分量はできるだけ多くしてください。では、深呼吸をして回答に取り組んでください。\n
        \n
        """)
        return long_templates

    raise ValueError(f"Invalid language {language}")

def streamlit_title_long_review_papers(papers, query_text, model = 'gpt-4-32k', language="English"):
    abstracts = []
    linked_apas = []
    titles = []
    prompts = title_long_review_generate_prompt([], query_text, language)
    caption = f"{len(papers)} 件の論文がレビューに使用されました。"
    for i, (title, abstract, year, link) in enumerate(zip(papers["title"], papers["abstract"], papers['year'], papers['linked_APA']), 1):
        text = f"[{i}] (Published in {year}) {title}\n{abstract}"
        abstracts.append(text)
        linked_apas.append(f"[{i}] : {link}")
        titles.append(f"[{i}] : {title}\n{abstract}")

        prompts = title_long_review_generate_prompt(abstracts, query_text, language)
        for prompt in prompts:
            if not is_valid_tiktoken(model, prompt):
                prompts = title_long_review_generate_prompt(abstracts[:-1], query_text, language)
                caption = f"{i - 1} / {len(papers)} 件の論文がレビューに使用されました。"
                break
    result = st.empty()
    cluster_summary = ""
    if language == '日本語':
        sections = ['## 定義', '## 要約', '## 最近の発展', '## 未解決の問題', '## 手法', '## 薬剤']
    elif language == 'English':
        sections = ['## Definition', '## Summary', '## Recent Developments', '## Outstanding Issues', '## Methods', '## Drugs']
    else:
        raise ValueError(f'Invalid language {language}')
    for section, prompt in zip(sections, prompts):
        cluster_summary +=  '\n' + section + '\n\n'
        result.write('\n' + section + '\n\n')
        for response in get_azure_gpt_response_stream(prompt, model):
            response_text = response.choices[0].delta.content
            if response_text:
                cluster_summary += response_text
                result.write(cluster_summary)
    result.empty()
    return cluster_summary, linked_apas, caption, titles

def title_review_generate_prompt(abstracts, query_text, language):
    abstracts_text = "\n\n".join(abstracts)

    if language == "English":
        # Query:\n\n{query_text}を Academic Abstracts のあとに入れる？
        prompt = f"""You are a researcher working on a literature review. The following are abstracts of academic papers you have gathered for the review.\n\n
                    {abstracts_text}
                    \n
                    Based on these, please perform the following tasks to create a literature review on {query_text}. Ignore abstracts that are not related to {query_text}.
                    Do not use prior knowledge or assumptions, and answer with lateral thinking.\n
                    \n
                    ## Definition\n
                    Please define {query_text} within 1000 characters. If available, also mention demographics and prevalence.\n
                    \n
                    ## Summary\n
                    Organize the content of the given abstracts and describe them comprehensively within 1000 characters. Cover all important points and main ideas presented in the original text, while condensing the information into a concise and understandable format. Avoid unnecessary information and repetition, and include relevant details and examples that support the main ideas. The length should be sufficiently long, providing a clear and accurate summary without omitting important information.\n
                    \n
                    ## Recent Developments\n
                    Discuss recent developments in this research field within 1000 characters, considering the publication year indicated in (Published in). Place particular emphasis on developments within the last five years.\n
                    Cover all important points and main ideas presented in the abstracts, while condensing the information into a concise and understandable format. Avoid unnecessary information and repetition, and include relevant details and examples that support the main ideas. The length should be sufficiently long, providing a clear and accurate summary without omitting important information.\n
                    \n
                    ## Unresolved Issues\n
                    List and explain in detail the unresolved issues in the research field pointed out in these abstracts, within 1000 characters for each item.\n
                    Information on unresolved issues should be as comprehensive as possible. If you cannot reflect all information from the abstracts, state 'References to other unresolved issues are also found in other abstracts ([number][number][number]...)' and indicate the [number] of those abstracts.\n
                    -
                    -
                    -
                    \n
                    ## Methods\n
                    List and explain in detail the methods and treatments used in this research field and their effects, as pointed out in these abstracts, within 1000 characters for each item.\n
                    Information on methods and treatments should be as comprehensive as possible. If you cannot reflect all information from the abstracts, state 'References to other methods/treatments are also found in other abstracts ([number][number][number]...)' and indicate the [number] of those abstracts.\n
                    -
                    -
                    -
                    \n
                    ## Drugs\n
                    List and explain in detail the drugs used in this research field and their safety and efficacy, as pointed out in these abstracts, within 1000 characters for each item.\n
                    Information on drugs should be as comprehensive as possible. If you cannot reflect all information from the abstracts, state 'References to other drugs are also found in other abstracts ([number][number][number]...)' and indicate the [number] of those abstracts.\n
                    -
                    -
                    -
                    \n
                    In your answer, please cite using the format [number]. If the abstract mentions OpenAccess, write it as [number OpenAccess]. Please answer in Japanese.\n
                    Try to make your response as lengthy as possible. Now, take a deep breath and start answering.\n"""
        return prompt

    if language == "日本語":
        japanese_prompt = f"""あなたは文献レビューに取り組む研究者です。以下は文献レビューの対象として取得した学術論文のアブストラクトです。\n\n
                            {abstracts_text}
                            \n
                            これらに基づいて、{query_text}についての文献レビューを作成するための以下のタスクを実行してください。
                            {query_text}と関連しないアブストラクトは無視してください。
                            なお、事前知識や思い込みは使用しないで、水平思考で答えてください。\n
                            \n
                            ## 定義\n
                            {query_text}について1000字以内で定義してください。情報があれば人口動態や罹患率についても述べてください。\n
                            \n
                            ## 要約\n
                            与えられたアブストラクトの内容を整理して、1000字以内で包括的に記述してください。
                            原文で提示されている重要なポイントや主要なアイデアをすべてカバーすると同時に、情報を簡潔で理解しやすい形式に凝縮する必要があります。
                            不必要な情報や繰り返しを避けながら、主要なアイデアを裏付ける関連する詳細や例を含めるようにしてください。
                            長さは十分に長くであるべきで、重要な情報を省略することなく、明確で正確な概要を提供すること。\n
                            \n
                            ## 最近の発展\n
                            この研究分野の最近の発展について（Published in）に示される出版年を参考に1000字以内で述べてください。特に5年以内の発展を重視すること。\n
                            アブストラクトで提示されている重要なポイントや主要なアイデアをすべてカバーすると同時に、情報を簡潔で理解しやすい形式に凝縮する必要があります。
                            不必要な情報や繰り返しを避けながら、主要なアイデアを裏付ける関連する詳細や例を含めるようにしてください。
                            文章の長さは、十分に長くあるべきで、重要な情報を省略することなく、明確で正確な概要を提供すること。\n
                            \n
                            ## 未解決の問題\n
                            これらのアブストラクトで指摘されている研究分野の未解決の問題をリストアップして、各項目について1000字以内で詳細に説明してください。\n
                            未解決の問題に関する情報は包括的であればあるほど望ましいです。
                            もしアブストラクトの情報をすべて反映できない場合には「他の未解決の問題についての言及も他のアブストラクトの中にあります（[number][number][number]...）」とそのアブストラクトの[number]を示して述べてください。\n
                            -
                            -
                            -
                            \n
                            ## 手法\n
                            これらのアブストラクトで指摘されているこの研究分野で用いられている手法や治療法とその効果をリストアップして、各項目について1000字以内で詳細に説明しtください。\n
                            手法や治療法の情報は包括的であればあるほど望ましいです。
                            もしアブストラクトの情報をすべて反映できない場合には「他の手法・治療法についての言及も他のアブストラクトの中にあります（[number][number][number]...）」とそのアブストラクトの[number]を示して述べてください。\n
                            -
                            -
                            -
                            \n
                            ## 薬剤\n
                            これらのアブストラクトで指摘されているこの研究分野で用いられている薬剤とその安全性・有効性をリストアップして、各項目について1000字以内で詳細に説明してください。\n
                            薬剤の情報は包括的であればあるほど望ましいです。もしアブストラクトの情報をすべて反映できない場合には「他の薬剤についての言及も他のアブストラクトの中にあります（[number][number][number]...）」とそのアブストラクトの[number]を示して述べてください。\n
                            -
                            -
                            -
                            \n
                            回答にあたっては、[number]という形式で引用を明記してください。アブストラクトにOpenAccessと明記されている場合には[number OpenAccess]と記述してください。日本語で回答してください。\n
                            文章の分量はできるだけ多くしてください。では、深呼吸をして回答に取り組んでください。\n"""
        return japanese_prompt

    raise ValueError(f"Invalid language {language}")


def streamlit_title_review_papers(papers, query_text, model = 'gpt-4-32k', language="English"):
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
    result = st.empty()
    cluster_summary = ""
    for response in get_azure_gpt_response_stream(prompt, model):
        response_text = response.choices[0].delta.content
        if response_text:
            cluster_summary += response_text
            result.write(cluster_summary)
    result.empty()
    return cluster_summary, linked_apas, caption, titles


#論文のリバイズ部分
def summary_writer_generate_prompt(references, cluster_summary, draft, language, mode):
    references_text = "\n\n".join(references)

    if mode == "revise_and_add_citation":
        prompt = f"""Specific summary: \n\n 
                    {cluster_summary}\n\n 
    
                    References:\n\n
                    {references_text}\n\n
    
                    Instructions: Using the provided academic summaries, write a comprehensive long description about the given draft by synthesizing these summaries. You must write in English in Markdown format.
                    Make sure to cite results using [number] notation after the sentence. If the provided search results refer to multiple subjects with the same name, 
                    write separate answers for each subject.\n\n
    
                    Draft: {draft}\n\n
                    """

        japanese_prompt = f"""学術論文の要約：\n\n
                    {cluster_summary}\n\n
    
                    参考文献： \n\n
                    {references_text}\n\n
    
                    指示: 提供された学術論文の要約を総合して、与えられた草稿について包括的な長文の説明を記述しなさい。必ず Markdown 形式の日本語で記述しなさい。
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

def streamlit_summery_writer_with_draft(cluster_summary, draft, references, model = 'gpt-4-32k',language="English", mode="only_add_citation"):
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

    result = st.empty()
    summary = ""
    for response in get_azure_gpt_response_stream(prompt, model):
        response_text = response.choices[0].delta.content
        if response_text:
            summary += response_text
            result.write(summary)
    result.empty()
    return summary, caption

def japanese_abstract_generate_prompt(abstract):
    japanese_prompt = f"""abstract: \n\n
                {abstract}
                \n\n
                指示: 上記の abstract を一切省略せずに日本語に翻訳して出力する。\n\n
                """
    return japanese_prompt

def gpt_japanese_abstract(abstract, model = 'gpt-3.5-turbo-1106'):
    prompt = japanese_abstract_generate_prompt(abstract)
    if is_valid_tiktoken(model, prompt):
        try:
            result = st.sidebar.empty()
            japanese_abstract = ""
            for response in get_azure_gpt_response_stream(prompt, model):
                response_text = response.choices[0].delta.content
                if response_text:
                    japanese_abstract += response_text
                    result.write(japanese_abstract)
            result.empty()
            return japanese_abstract
        except Exception as e:
            print(f"Error as {e}")
            return f"エラーが発生しました。\n{e}"
    else:
        return "アブストラクトの長さが規定の長さよりも長いため送信できませんでした。"


def japanese_peper_interpreter_generate_prompt(pdf_text):
    japanese_prompt = f"""
            以下は学術論文の本文です。この内容に従って以下のタスクを実行してください。
            
            {pdf_text}
            
            タスク：この論文の内容を以下のセクションごとに包括的に記述してください。事前知識やここに書かれていない知識は用いないでください。
            本文で提示されている重要なポイントや主要なアイデアをすべてカバーしてください。主要なアイデアを裏付ける関連する詳細や例を含めるようにしてください。
            長さは十分に長くであるべきで、重要な情報を省略することなく、明確で正確な概要を提供すること。
            
            ## Title
            
            ## Abstract
            
            ## Introduction (Background)
            
            ## Related Works and Unsolved Problems (Research Gap)
            
            ## Theoretical Perspective
            
            ## Research Questions and Hypotheses
            
            ## Method
            
            ## Results
            
            ## Discussion
            
            ## Conclusion
            
            ## Application
            
            ## Treatments
            
            ## Medicine
            
            ## Limitation
            
            回答は日本語で行ってください。
            
            """
    return japanese_prompt

def japanese_paper_chat(pdf_text, chat_log, model = 'gpt-4-1106-preview'):
    system_prompt = f"以下は学術論文の本文です。この内容に従って最後のユーザの質問に答えてください。{pdf_text}\n\n回答は日本語で行ってください。\n\n"
    messages = [{"role" : "system", "content" : system_prompt}]
    for chat in chat_log:
        messages.append({"role" : chat['role'], "content": chat['content']})

    if is_valid_tiktoken(model, system_prompt):
        try:
            chat_message = st.sidebar.empty()
            assistant_response = ""
            for response in get_azure_gpt_message_stream(messages, model):
                response_text = response.choices[0].delta.content
                if response_text:
                    assistant_response += response_text
                    chat_message.write(assistant_response)
            chat_message.empty()
            return assistant_response
        except Exception as e:
            print(f"Error as {e}")
            return f"エラーが発生しました。\n{e}"
    else:
        return "論文の長さが規定の長さよりも長いため送信できませんでした。"


def gpt_japanese_paper_interpreter(abstract, model = 'gpt-4-1106-preview'):
    prompt = japanese_peper_interpreter_generate_prompt(abstract)
    if is_valid_tiktoken(model, prompt):
        try:
            result = st.sidebar.empty()
            japanese_abstract = ""
            for response in get_azure_gpt_response_stream(prompt, model):
                response_text = response.choices[0].delta.content
                if response_text:
                    japanese_abstract += response_text
                    result.write(japanese_abstract)
            result.empty()
            return japanese_abstract
        except Exception as e:
            print(f"Error as {e}")
            return f"エラーが発生しました。\n{e}"
    else:
        return "論文の長さが規定の長さよりも長いため送信できませんでした。"



def construct_direct_quotations_graph(papers_df):
    # 直接引用法によるネットワークを作成するためにGraphの初期化
    G = nx.Graph()
    # papers_df DataFrameをループして処理
    for k in range(len(papers_df)):
        paper_id = papers_df.loc[k, 'paperId']
        paper_year = papers_df.loc[k, 'year']
        # ノードの追加条件をチェック
        if pd.notnull(paper_id) and pd.notnull(paper_year):  # pd.notnullを使用してnull値をチェック
            G.add_node(paper_id, title=f"From:{papers_df.loc[k, 'title']}", citationCount=papers_df.loc[k, 'citationCount'],
                       year=paper_year)
        else:
            continue  # paper_id または paper_year が null の場合はノードの追加をスキップ

        # paperが引用している文献についての処理
        for reference in papers_df.loc[k, 'references']:
            ref_id = reference.get('paperId')
            ref_year = reference.get('year')
            ref_citeCount = reference.get('citationCount', 0)
            if pd.notnull(ref_id) and pd.notnull(ref_year) and pd.notnull(ref_citeCount):  # 参照のpaperIdとyearもチェック
                # 参照がすでにノードとして追加されていない場合にのみ追加
                if ref_id not in G:
                    G.add_node(ref_id, title=f"To:{reference['title']}",
                               citationCount=reference.get('citationCount', 0), year=ref_year)
                # エッジの追加（重みはu,vの被引用回数の平均）
                G.add_edge(paper_id, ref_id,
                           weight=(papers_df.loc[k, 'citationCount'] + ref_citeCount) / 2)

        # paperを引用している文献についての処理
        for citation in papers_df.loc[k, 'citations']:
            cite_id = citation.get('paperId')
            cite_year = citation.get('year')
            cite_citeCount = citation.get('citationCount', 0)
            if pd.notnull(cite_id) and pd.notnull(cite_year) and pd.notnull(cite_citeCount):  # 参照のpaperIdとyearもチェック
                # 参照がすでにノードとして追加されていない場合にのみ追加
                if cite_id not in G:
                    G.add_node(cite_id, title=f"To:{citation['title']}",
                               citationCount=cite_citeCount, year=cite_year)
                # エッジの追加（重みはu,vの被引用回数の平均）
                G.add_edge(paper_id, cite_id,
                           weight=(papers_df.loc[k, 'citationCount'] + cite_citeCount) / 2)
    # 無向グラフにする
    H = G.to_undirected()
    return H, G

def construct_direct_quotation_and_scrivener_combination(papers_df, threshold_year, threshold_degree):
    _, G = construct_direct_quotations_graph(papers_df)
    # thres_year以降のノードのみを含む辞書を作成
    neighbors_dict = {node: set(G.neighbors(node)) for node in G.nodes() if G.nodes[node].get('year', 0) >= threshold_year}

    # エッジを追加するためのリストを作成
    edges_to_add = []

    # フィルタリングされたノードの組み合わせを生成
    for u, v in itertools.combinations(neighbors_dict.keys(), 2):
        # 事前計算されたリストを使用して共通の隣接ノードを見つける
        common_neighbors = neighbors_dict[u] & neighbors_dict[v]
        common_count = len(common_neighbors)

        # 共通の隣接ノードが1つ以上ある場合にエッジを追加
        if common_count > 0:
            u_citation_count = G.nodes[u].get('citationCount', 0)
            v_citation_count = G.nodes[v].get('citationCount', 0)
            average_citation_count = (u_citation_count + v_citation_count) / 2
            edges_to_add.append((u, v, {'weight': average_citation_count}))

    # 一度にすべてのエッジを追加
    G.add_edges_from(edges_to_add)
    # グラフの次数が1より大きいノードだけを取得（1回以上引用されている）
    large_degree = [node for node, degree in dict(G.degree()).items() if degree > threshold_degree]
    # サブグラフの作成
    H = G.subgraph(large_degree)
    # # 無向グラフにする
    # H = G.to_undirected()
    return H, G

def generate_windows(start_year, end_year, threshold_year ,window_size):
    """
    指定された開始年、終了年、ウィンドウサイズに基づいてウィンドウのリストを生成する関数。
    """
    windows = []

    # 過去10年(thresholdhold_year)よりも前はひとつのウィンドウにまとめてしまう
    if start_year < threshold_year:
        windows.append((start_year, threshold_year - 1))
    current_start = threshold_year

    # 過去10年間をウィンドウサイズごとにwindowsに追加
    while current_start <= end_year:
        current_end = min(current_start + window_size - 1, end_year)
        windows.append((current_start, current_end))
        current_start += window_size
        # 最後のウィンドウが単一年であれば、それをそのまま使用
        if current_start > end_year:
            if current_end == end_year:
                break

    return windows

# @st.cache_data
def get_cluster_papers(df_centrality, cluster_nodes):
    df_centrality = df_centrality.copy()
    df_centrality_filtered = df_centrality[df_centrality['Node'].isin(cluster_nodes)]
    order_dict = {node: index for index, node in enumerate(cluster_nodes)}
    # Adding a temporary column for sorting
    df_centrality_filtered ['sort_order'] = df_centrality_filtered ['Node'].map(order_dict)
    # Sorting by the temporary column and dropping it
    df_centrality_filtered = df_centrality_filtered .sort_values(by='sort_order').drop(columns=['sort_order'])
    df_centrality_filtered['Title'] = df_centrality_filtered['Title'].apply(lambda x: x.replace('To:', '').replace('From:', ''))
    return df_centrality_filtered

#すべての論文を検索順に並べ替えるコード
def sort_ids_by_reference_order(reference_ids, ids_to_sort):
    # Creating a set for faster lookup
    reference_set = set(reference_ids)
    # Filter out ids that are not in the reference list
    filtered_ids = [id_ for id_ in ids_to_sort if id_ in reference_set]
    # Sort based on the index in the reference list
    sorted_ids = sorted(filtered_ids, key=lambda x: reference_ids.index(x))
    return sorted_ids

# @st.cache_resource
def cluster_for_year(_H, df_centrality, start_year, end_year, threshold_year):
    windows = generate_windows(min(start_year, df_centrality['Year'].min()), max(end_year, df_centrality['Year'].max()), threshold_year, 2)
    clustering = []

    for _start_year, _end_year in windows:
        # サブグラフの抽出
        subgraph = _H.subgraph([n for n, attr in _H.nodes(data=True) if _start_year <= attr['year'] <= _end_year])

        if len(subgraph) == 0:
            continue

        # Louvain法によるクラスタリング
        partition = community_louvain.best_partition(subgraph, random_state=42)

        # ノード数が1のクラスタはひとつにまとめる
        # 各クラスタのノード数をカウント
        cluster_counts = defaultdict(int)
        for node, cluster in partition.items():
            cluster_counts[cluster] += 1

        # ノード数が1のクラスタを特定し、'residuals'クラスタに割り当てる
        residuals_cluster = max(cluster_counts.keys()) + 1  # 新しいクラスタ番号
        for node, cluster in partition.items():
            if cluster_counts[cluster] == 1:
                partition[node] = residuals_cluster

        # データフレームにクラスタ情報を追加
        for node, cluster in partition.items():
            cluster_id = str(cluster + 1).zfill(2)
            clustering.append({
                'Node': node,
                'Cluster': f"〜{_end_year}-{cluster_id}"
            })

    # クラスタリング結果をデータフレームに変換
    clustering = pd.DataFrame(clustering)
    # df_centralityにCluster列をマージ
    df_centrality = pd.merge(df_centrality, clustering, on='Node')
    return df_centrality


#全体の df_centralityの計算
def page_ranking_sort(H, start_year, end_year, threshold_year):
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

    df_centrality = cluster_for_year(H, df_centrality, start_year, end_year, threshold_year)

    return df_centrality



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
                initials = ". ".join([name[0] for name in parts[:-1] if len(name) > 0])
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



# Y座標を少しランダムにずらすための関数を追加します
def randomize_y_position(start, end, excluded, count, seed=42):
    random.seed(seed)
    ys = []
    while len(ys) < count:
        y = random.uniform(start, end)
        if y not in excluded:
            ys.append(y)
            excluded.add(y)
    return ys


# `edge_data`関数を修正して`G`を含むようにする
def edge_data(B, C, G):
    # この内部関数は`G`を閉じ込めて、BとCだけを外部から受け取れるようにする
    def edge_data_with_graph(B, C):
        edge_count = sum(1 for node_in_b in B for node_in_c in C if G.has_edge(node_in_b, node_in_c))
        return {'weight': edge_count}

    return edge_data_with_graph(B, C)


# @st.cache_resource
def edge_to_curved_trace(_edge, _pos, _width, curvature=0.1):
    # エッジの開始と終了の点を取得します
    x0, y0 = _pos[_edge[0]]
    x1, y1 = _pos[_edge[1]]

    # 曲線を滑らかにするために制御点を計算します
    # ここでは、エッジの中間点を制御点として単純化しています
    # 実際には、より複雑な制御が必要な場合もあります
    ctrl_x, ctrl_y = (x0 + x1) / 2, (y0 + y1) / 2 + curvature

    # 曲線のポイントを生成するためにベジェ曲線の式を使います
    bezier_points = bezier_curve(np.array([x0, ctrl_x, x1]), np.array([y0, ctrl_y, y1]), 20)

    edge_trace = go.Scatter(
        x=bezier_points[:, 0],
        y=bezier_points[:, 1],
        line=dict(width=_width, color='rgba(136, 136, 136, 0.6)'),
        hoverinfo='none',
        mode='lines'
    )

    return edge_trace


def bezier_curve(points_x, points_y, num_of_points):
    # ベジェ曲線のポイントを計算する関数
    n = len(points_x) - 1
    return np.array([sum(
        [bernstein_poly(i, n, t) * np.array([x, y]) for i, (x, y) in enumerate(zip(points_x, points_y))])
        for t in np.linspace(0, 1, num_of_points)])


def bernstein_poly(i, n, t):
    # ベルンシュタイン多項式を計算
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

# @st.cache_data
def process_display_cluster(df_centrality, cluster_df):
    display_cluster = df_centrality.copy()
    display_cluster['Year'] = display_cluster['Year'].apply(lambda x: int(str(x).split('.')[0]))
    cluster_df = cluster_df.copy()
    cluster_df['Year'] = cluster_df['Year'].apply(lambda x: int(str(x).split('.')[0]))
    return display_cluster, cluster_df


# @st.cache_resource
def create_quotient_graph(_display_cluster, _H , _communities, _partition):
    # dfを使用してcommunities辞書を作成（キー：クラスタ番号、値：所属ノードのリスト）
    # 新しいグラフ B を作成
    B = nx.Graph()

    # communities の各コミュニティをノードとして追加
    for comm_id in _communities.keys():
        B.add_node(comm_id)

        # コミュニティ間のエッジを追加
    for comm_id, nodes in _communities.items():
        # 当該コミュニティ内のノードについてループ
        for node in nodes:
            # ノードの隣接ノードについてループ
            for neighbor in _H.neighbors(node):
                neighbor_comm_id = _partition[neighbor]
                # 異なるコミュニティに属する場合にのみ
                if comm_id != neighbor_comm_id:
                    # 既にエッジが追加されているか確認
                    if B.has_edge(comm_id, neighbor_comm_id):
                        # エッジが存在する場合は重みを増やす
                        B[comm_id][neighbor_comm_id]['weight'] += 1
                    else:
                        # エッジが存在しない場合は追加し、重みを1に設定
                        B.add_edge(comm_id, neighbor_comm_id, weight=1)
    return B

# @st.cache_data
def process_communities(display_cluster, cluster_df, _H, _communities, _partition):
    block_model_graph = create_quotient_graph(display_cluster, _H, _communities, _partition)
    sorted_communities_by_year = cluster_df.sort_values('Year').index.tolist()
    y_positions = randomize_y_position(-1, 1, set(), len(sorted_communities_by_year))
    pos = {node: (index, y) for index, (node, y) in enumerate(zip(sorted_communities_by_year, y_positions))}
    pos = {node: pos[node] for node in block_model_graph.nodes() if node in pos}
    return block_model_graph,  pos

# @st.cache_resource
def process_edges(_block_model_graph, pos):
    weights = nx.get_edge_attributes(_block_model_graph, 'weight').values()
    max_weight = max(weights) if weights else 1
    widths = [w * 5.0 / max_weight for w in weights]  # 重みに比例するようにエッジの太さを設定

    edge_traces = []  # エッジのための複数のtraceを保持するリストを作成

    for (edge, width) in zip(_block_model_graph.edges(), widths):
        edge_trace = edge_to_curved_trace(edge, pos, width, curvature=0.4)
        edge_traces.append(edge_trace)  # 新しいtraceをリストに追加

    return edge_traces, weights

# @st.cache_resource
def process_sizes(_block_model_graph, cluster_df):
    min_citations = cluster_df['Node'].min()
    max_citations = cluster_df['Node'].max()
    size_base = 10
    min_size = size_base
    max_size = size_base * 10
    # ブロックモデルグラフのノードリストを取得
    block_nodes = list(_block_model_graph.nodes)
    sizes = [((citationCount - min_citations) / (max_citations - min_citations) * (max_size - min_size) + min_size)
             if max_citations > min_citations else min_size for citationCount in cluster_df.loc[block_nodes, 'Node']]

    return sizes, block_nodes

# @st.cache_resource
def process_nodes(_block_model_graph, pos, cluster_df):
    # ノードの情報を抽出
    node_x = []
    node_y = []
    node_text = []

    for node in _block_model_graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # 'Keywords'列からホバーテキストを生成
        node_text.append(
            f"{node}: {int(cluster_df.loc[node, 'Year'])}年({cluster_df.loc[node, 'Recent5YearsCount']}/{cluster_df.loc[node, 'Node']}) ({cluster_df.loc[node, 'CitationCount']}) {cluster_df.loc[node, 'ClusterKeywords']}")  # gbがクラスタの情報を含むDataFrameと仮定
    return   node_x, node_y, node_text

def create_plot(sizes, block_nodes, node_x, node_y, node_text,  edge_traces):

    # クラスタ番号を表示するための追加のScatterトレースを作成
    cluster_number_trace = go.Scatter(
        x=node_x, y=node_y,
        text=block_nodes,  # クラスタ番号を含むテキストリスト
        mode='text',  # テキストのみを表示
        hoverinfo='none',  # ホバー情報は不要
        textposition="top center",  # テキストの位置をノードの上に設定
        showlegend=False,  # 凡例には表示しない
    )

    # ノードのサイズと色のリスト
    # sizes というリストがノードごとのサイズを持っていると仮定する
    node_color = 'lightsteelblue'  # またはノードごとに異なる色のリストを指定する

    # ノードのScatterトレースを作成
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_text,  # 以前に作成したノードのラベルリスト
        mode='markers',  # マーカーとテキストの両方を表示
        hoverinfo='text',
        marker=dict(
            showscale=False,  # これをFalseに設定してカラーバーを非表示にする
            color=node_color,
            size=sizes,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2),
        textposition="bottom center"  # ラベルの位置
    )


    # レイアウトの作成
    layout = go.Layout(
        title='クラスタの時間的な発展',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),  # 右側のマージンを増やしてグラフを左に移動
        annotations=[dict(
            text="円の大きさ:論文数, 辺の太さ:クラスタの関連性, X軸:出版年の新しさ, 注釈:[平均出版年]（5年以内の論文数/総論文数）(平均被引用回数)[キーワード]",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        width=1200,
    )

    # フィギュアの作成
    fig = go.Figure(data=[node_trace, cluster_number_trace] + edge_traces, layout=layout)

    # Streamlitでのプロットの表示
    st.plotly_chart(fig)

# @st.cache_data
def plot_research_front(_df_centrality, _H, _cluster_df, _cluster_id_paper_ids, _partition):
    # 元の関数の実行部分
    display_cluster, cluster_df = process_display_cluster(_df_centrality, _cluster_df)
    block_model_graph,  pos = process_communities(display_cluster, cluster_df, _H, _cluster_id_paper_ids, _partition)
    edge_traces, weights = process_edges(block_model_graph, pos)
    sizes, block_nodes  = process_sizes(block_model_graph, cluster_df)
    node_x, node_y, node_text = process_nodes(block_model_graph, pos, cluster_df)
    create_plot(sizes, block_nodes, node_x, node_y, node_text,  edge_traces)

# PDFをダウンロードする関数
def download_pdf(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'referer': 'https://www.google.com/',
            'Accept': 'application/pdf',
        }

        response = requests.get(url, headers= headers)

    except Exception as e:
        print(f"Error happened as {e}")
        return None
    if response.status_code != 200:
        print(f"Error: HTTP status code {response.status_code}")
        return None

    filename = url.split('/')[-1]
    file_path = os.path.join(current_dir, "database", filename)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return filename

# PDFの内容をヘッダーとフッターを無視してテキストとして抽出する関数
def extract_text_from_doc(doc):
    text = ""
    for page in doc:
        # ページの寸法を取得
        rect = page.rect
        # ヘッダーとフッターの高さを定義
        header_height = 50
        footer_height = 50
        # ページの本文のみを抽出するための矩形を定義
        content_rect = fitz.Rect(rect.x0, rect.y0 + header_height, rect.x1, rect.y1 - footer_height)
        # 定義した矩形内のテキストのみを抽出
        text += page.get_text("text", clip=content_rect)
    return text

def extract_text_without_headers_footers(pdf_file):
    pdf_path = os.path.join(data_folder, pdf_file)
    with fitz.open(pdf_path, filetype="pdf") as doc:
        text = extract_text_from_doc(doc)
    os.remove(pdf_path)
    return text

def extract_text_without_headers_footers_from_stream(stream):
    with fitz.open(stream=stream.read(), filetype="pdf") as doc:
        text = extract_text_from_doc(doc)
    return text



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
    refresh_button = st.button("REFRESH")
    if refresh_button:
        st.rerun()