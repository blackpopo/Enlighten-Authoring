import requests
from collections import defaultdict, namedtuple
import datetime
import numpy as np
import japanize_matplotlib
from persist import persist, load_widget_state
import time
import json
import streamlit as st
import ast
import pandas as pd
import matplotlib.pyplot as plt

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
DEEPL_API_KEY = st.secrets['DEEPL_API_KEY']
SemanticScholar_API_KEY = st.secrets['SEMANTICSCHOLAR_API_KEY']


st.set_page_config(
    page_title="Enlighten Authoring",
    initial_sidebar_state="auto")

def display_dataframe(df, title, topk, columns=None):
    st.write(
        f"<h5 style='text-align: left;'> {title} </h5>",
        unsafe_allow_html=True,
    )
    # st.subheader(title)
    if columns != None:
        df = df[columns]
    st.dataframe(df.head(topk))

def display_cluster_dataframe(df, title, topk):
    st.write(
        f"<h5 style='text-align: left;'> {title} </h5>",
        unsafe_allow_html=True,
    )
    # st.subheader(title)
    def get_journal_name(journal_info):
        try:
            if isinstance(journal_info, str):
                # journal_infoを辞書に変換し、'name'キーの値を取得します
                journal_info = ast.literal_eval(journal_info)
            return journal_info.get('name', '')
        except Exception as e:
            # 文字列が辞書として評価できない場合やキーが存在しない場合は空文字列を返します
            # print(f'get journal name error {e}')
            return ""

    # 'journal'列の各エントリにget_journal_name関数を適用して新しい列を作成します
    df['journal name'] = df['journal'].apply(get_journal_name)
    df['author names'] = df['authors'].apply(lambda x: [d.get('name') for d in x] if isinstance(x, list) else None)
    df['citation count'] = df['citationCount']
    df['published year'] = df['year'].apply(lambda x: str(x).replace('.0', ''))

    if not 'japanese abstract' in df.columns:
        df = df[['Title', 'Importance', 'abstract', 'published year', 'citation count', 'journal name', 'author names']]
        df.columns = ['Title', 'Importance', 'Abstract', 'Published Year', 'Citation Count', 'Journal Name', 'Author Names']
    else:
        df = df[['Title', 'Importance', 'japanese abstract', 'abstract', 'published year', 'citation count', 'journal name', 'author names']]
        df.columns = ['Title', 'Importance', 'Japanese Abstract', 'Abstract', 'Published Year', 'Citation Count', 'Journal Name', 'Author Names']
    st.dataframe(df.head(topk), hide_index=True)

def display_dataframe_detail(df, title, topk):
    st.write(
        f"<h5 style='text-align: left;'> {title} </h5>",
        unsafe_allow_html=True,
    )
    # st.subheader(title)
    def get_journal_name(journal_info):
        try:
            if isinstance(journal_info, str):
                # journal_infoを辞書に変換し、'name'キーの値を取得します
                journal_info = ast.literal_eval(journal_info)
            return journal_info.get('name', '')
        except Exception as e:
            # 文字列が辞書として評価できない場合やキーが存在しない場合は空文字列を返します
            return ""

    # 'journal'列の各エントリにget_journal_name関数を適用して新しい列を作成します
    df['journal name'] = df['journal'].apply(get_journal_name)
    df['author names'] = df['authors'].apply(lambda x: [d.get('name') for d in x] if isinstance(x, list) else None)
    df['citation count'] = df['citationCount']
    df['published year'] = df['year'].apply(lambda x: str(x))
    if not 'japanese_abstract' in df.columns:
        df = df[['title', 'abstract', 'published year', 'citation count', 'journal name', 'author names']]
        df.columns =  ['Title',  'Abstract', 'Published Year', 'Citation Count', 'Journal Name', 'Author Names']
    else:
        df = df[['title', 'japanese abstract','abstract', 'published year', 'citation count', 'journal name', 'author names']]
        df.columns =  ['Title', 'Japanese Abstract',  'Abstract', 'Published Year', 'Citation Count', 'Journal Name', 'Author Names']
    st.dataframe(df.head(topk), hide_index=True)

def display_title():
    st.title("Enlighten Authoring")
    st.markdown("<strong>臨床的位置づけ立案支援AI: 専門情報のレビューと文章へのエビデンスの付与を行います</strong>", unsafe_allow_html=True)


# def display_list(text_list, size=4):
#     if not isinstance(text_list, list):
#         st.write("文字列のリストを入力してください")
#         return
#
#     for text in text_list:
#         st.write(
#             f"<h{size} style='text-align: left;'> {text} </h{size}>",
#             unsafe_allow_html=True,
#         )


def display_description(description_text = 'This is application description.', size=5):
    """Displays the description of the app."""
    # st.markdown("<h4 style='text-align: left;'>Get answers to your questions from 200M+ research papers from Semantic Scholar, summarized by ChatGPT</h4>", unsafe_allow_html=True)
    description_text = description_text.replace('#', '\n')
    st.write(
        f"<h{size} style='text-align: left;'> {description_text} </h{size}>",
        unsafe_allow_html=True,
    )


def display_warning(warning_text = 'This is a warning text.'):
    """Displays a warning message in small text"""
    st.write(
        f"<h8 style='text-align: left;'>⚠️ Warning: {warning_text}</h8>️ ⚠️",
        unsafe_allow_html=True,
    )


def display_error(error_text = 'This is a error text.'):
    st.write(
        f"<h8 style='text-align: left;'>🚨 ERROR: {error_text}</h8>️ 🚨",
        unsafe_allow_html=True,
    )


def display_spaces(repeat=1):
    for _ in range(repeat):
        st.markdown("<br>", unsafe_allow_html=True)


def display_references_list(references_list, size=7):
    with st.expander('参考文献を表示'):
        for reference_text in references_list:
            st.write(
                f"<h{size} style='text-align: left;'> {reference_text} </h{size}>",
                unsafe_allow_html=True,
            )

def display_language_toggle(unique_string):
    toggle = st.radio(
        f"{unique_string}で使用する言語を選択してください。",
        [ '日本語', 'English']
    )
    return toggle

def display_draft_evidence_toggle(unique_string):
    toggle = st.radio(
        f"{unique_string}でエビデンスを付与する方法を選択してください。",
        [ '文献のみ付与', '文章の加筆と文献の付与']
    )
    if toggle == '文献のみ付与':
        toggle = "only_add_citation"
    elif toggle == '文章の加筆と文献の付与':
        toggle = "revise_and_add_citation"
    else:
        raise ValueError(f"Invalid toggle {toggle}")
    return toggle

def display_cluster_years(df: pd.DataFrame):
    display_description("クラスタ内の論文出版年", 5)
    min_year, max_year, ave_year = df['year'].min(), df['year'].max(), df['year'].mean()
    min_year, max_year, ave_year = str(min_year).replace('.0', ''), str(max_year).replace('.0', ''), str(np.around(ave_year, 2))
    current_year = datetime.datetime.now().year
    recent_5_years_count = df[df['year'] > (current_year - 5)].shape[0]
    display_description(f"直近 5 年に{recent_5_years_count}本の論文が出版されています。", 6)
    display_description(f"平均 {ave_year}年 ({min_year}年から{max_year}年)", 6)
    # Matplotlib でグラフ作成
    # 年ごとの論文数を計算
    paper_count_by_year = df['year'].value_counts().sort_index()

    # 横軸の設定
    start_year = int(paper_count_by_year.index.min())
    end_year = current_year
    if end_year - start_year >= 50:
        x_ticks_list = list(range(end_year, start_year, -5))[::-1]
        x_ticks_list[0] = start_year
        x_ticks = x_ticks_list  # 5年ごとの年をx_ticksに設定
    elif 50 > end_year - start_year  >= 10:
        x_ticks_list = list(range(end_year, start_year, -3))[::-1]
        x_ticks_list[0] = start_year
        x_ticks = x_ticks_list  # 5年ごとの年をx_ticksに設定
    else:
        x_ticks_list = list(range(end_year, start_year, -1))[::-1]
        x_ticks_list[0] = start_year
        x_ticks = x_ticks_list  # 5年ごとの年をx_ticksに設定

    # Matplotlib で折れ線グラフ作成
    plt.figure(figsize=(12, 6))
    plt.plot(paper_count_by_year.index, paper_count_by_year.values, color='aqua', marker='o')
    plt.xlabel('出版年')
    plt.ylabel('論文数[本]')
    plt.xticks(x_ticks, rotation=45)
    plt.yticks(np.arange(0, paper_count_by_year.values.max() + 1, step=1))

    # グラフの枠線を一番下の線以外消す
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # Streamlit でグラフ表示
    st.pyplot(plt)

def display_year_input():
    # 現在の年を取得
    current_year = datetime.datetime.now().year

    # 列を作成
    col1, col2 = st.columns(2)

    # 開始年と終了年を横並びに表示
    with col1:
        start_year = st.selectbox("検索の開始年", range(1880, current_year + 1), 0)

    with col2:
        end_year = st.selectbox("検索の終了年", range(1880, current_year + 1), current_year - 1880)

    st.session_state['year'] = f"{start_year}-{end_year}"


def get_session_info():
    # get session info
    session = requests.get("http://ip-api.com/json").json()
    return session


# Call the function to get the session info
def dump_logs(query, response, success=True):

    # session = get_session_info()
    # Create a dictionary of query details
    query_details = {
        # "session": session,
        "timestamp": time.time(),
        "query": query,
        "response": response,
    }

    if success:
        # Append the query details to a JSON file
        with open("query_details.json", "a") as f:
            json.dump(query_details, f)
            f.write("\n")
    else:
        # Append the query details to a JSON file
        with open("query_details_error.json", "a") as f:
            json.dump(query_details, f)
            f.write("\n")




def all_reset_session(session_state, except_key):
    for key in session_state:
        if key not in except_key:
            session_state.pop(key)
