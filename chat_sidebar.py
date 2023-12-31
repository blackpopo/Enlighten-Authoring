import numpy as np
import pandas as pd
import streamlit

from streamlit_utils import *
from utils import *

def set_sidebar_width():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 33.33%;
            max-width: 33.33%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_journal_name(journal_info):
    try:
        if isinstance(journal_info, str):
            # journal_infoを辞書に変換し、'name'キーの値を取得します
            journal_info = ast.literal_eval(journal_info)
        return journal_info.get('name', '')
    except Exception as e:
        # 文字列が辞書として評価できない場合やキーが存在しない場合は空文字列を返します
        return "ジャーナルの情報がありません。"

def apply_state_nan(open_access_info):
    if pd.isna(open_access_info):
        return np.nan


def display_selected_paper_component():
    cluster_df_detail = st.session_state['cluster_df_detail'].copy()
    cluster_df_detail = cluster_df_detail.reset_index()
    cluster_df_detail['published year'] = cluster_df_detail['year'].apply(lambda x: str(x).replace('.0', ''))
    cluster_df_detail["first author"] = cluster_df_detail["authors"].apply(lambda x: x[0]['name'] if len(x) > 0 else "")
    cluster_df_detail["journal name"] = cluster_df_detail["journal"].apply(get_journal_name)
    cluster_df_detail['open_access_pdf'] = cluster_df_detail['openAccessPdf'].apply(lambda x: get_open_access_info(x)[0] if x else np.nan)
    cluster_df_detail['is_open_access'] = cluster_df_detail['open_access_pdf'].apply(lambda x: False if not x else True)
    # キー用の新しい列を作成
    cluster_df_detail['options'] = (cluster_df_detail.index + 1).astype(str) + " " + \
                                   cluster_df_detail['first author'].astype(str) + " (" + \
                                   cluster_df_detail['published year'].astype(str) + ") " + \
                                   cluster_df_detail['title'].astype(str)
    # 辞書を作成
    papers_dict = cluster_df_detail.copy().set_index('options')[['Node', 'isOpenAccess']].to_dict(orient='index')
    # 各行の(Node, isOpenAccess)をタプルとして格納
    papers_dict = {k: {"node": v['Node'], "isOpenAccess": v['isOpenAccess']} for k, v in papers_dict.items()}

    selected_paper_node = st.sidebar.selectbox("詳細を表示したい論文を選択してください。", papers_dict.keys())

    if not 'chat_selected_paper_node' in st.session_state or st.session_state['chat_selected_paper_node'] != \
            papers_dict[selected_paper_node]["node"]:
        for key in st.session_state:
            if 'chat_' in key:
                st.session_state.pop(key)
        st.session_state["chat_selected_paper_node"] = papers_dict[selected_paper_node]["node"]
        st.session_state["chat_log"] = []

    selected_paper = cluster_df_detail[cluster_df_detail["Node"] == st.session_state['chat_selected_paper_node']]
    st.session_state['chat_selected_paper'] = selected_paper.iloc[0]

def display_paper_basic_information_component():
    if 'chat_selected_paper' in st.session_state:
        paper = st.session_state['chat_selected_paper']
        # 論文の基本情報を整理して表示
        if not paper.empty:
            # st.sidebar.write("論文タイトル:", paper['Title'])
            st.sidebar.write("論文タイトル" , f'<a href="https://www.semanticscholar.org/paper/{paper["paperId"]}" target="_blank">{paper["title"]}</a>', unsafe_allow_html=True)
            st.sidebar.write("著者:", paper['authors'][0]['name'] if len(paper['authors']) > 0 else "")
            st.sidebar.write("出版年:", paper['published year'])
            st.sidebar.write("引用数:", str(paper['CitationCount']))
            st.sidebar.write("クラスタ:", paper['Cluster'])
            st.sidebar.write("アブストラクト:", paper['abstract'] if pd.notna(paper['abstract']) else "アブストラクトがありません。")
            st.sidebar.write("ジャーナル:", paper['journal name'])
            st.sidebar.write("オープンアクセス:", "はい" if paper['isOpenAccess'] else "いいえ")
            # 必要に応じて他の情報も表示
        else:
            st.sidebar.write("選択された論文の情報が見つかりません。")

def display_japanese_abstract_component():
    if 'chat_selected_paper' in st.session_state:
        abstract  = st.session_state['chat_selected_paper']['abstract']
        # 論文とのチャット部分
        if not 'chat_japanese_abstract' in st.session_state and pd.notna(abstract):
            japanese_abstract_button = st.sidebar.button("日本語のアブストラクトを表示")
        else:
            japanese_abstract_button = False

        if japanese_abstract_button:
            # with st.sidebar.spinner("⏳ 日本語アブストラクトの取得中です..."):
            japanese_abstract = gpt_japanese_abstract(abstract)
            st.session_state['chat_japanese_abstract'] = japanese_abstract
            st.rerun()

        if 'chat_japanese_abstract' in st.session_state:
            st.sidebar.write("日本語のアブストラクト")
            st.sidebar.write(st.session_state['chat_japanese_abstract'])


def get_open_access_info(open_access_pdf):
    if pd.isna(open_access_pdf):
        return None, None
    if isinstance(open_access_pdf, str):
        try:
            open_access_pdf = ast.literal_eval(open_access_pdf)
            if "url" in open_access_pdf:
                pdf_url = open_access_pdf["url"]
                status = open_access_pdf['status']
            else:
                return None, None
        except Exception as e:
            print(f'Error in decode open access url {e}')
            return None, None
    else:
        pdf_url = open_access_pdf['url']
        status = open_access_pdf['status']
    return pdf_url, status

def display_open_access_paper_information_component():
    if 'chat_selected_paper' in st.session_state:
        selected_paper = st.session_state['chat_selected_paper']

        if 'chat_is_open_access' in st.session_state:
            is_open_access = st.session_state['chat_is_open_access']
        else:
            is_open_access = selected_paper['isOpenAccess']

        if is_open_access:
            pdf_button = st.sidebar.button("PDF を取得")
            if pdf_button:
                pdf_url = selected_paper['open_access_pdf']
                if pdf_url:
                    pdf_path = download_pdf(pdf_url)
                else:
                    pdf_path = None
                if not pdf_path:
                    st.sidebar.write("PDF が取得できませんでした。ファイルをアップロードしてください。")
                    is_open_access = False
                else:
                    try:
                        pdf_text = extract_text_without_headers_footers(pdf_path)
                        st.sidebar.write("PDF を取得しました。")
                        st.session_state['chat_pdf_text'] = pdf_text
                        print(f'pdf text length {len(pdf_text)}')
                    except Exception as e:
                        st.sidebar.write("PDF が取得できませんでした。ファイルをアップロードしてください。")
                        is_open_access = False

        if not is_open_access:
            st.sidebar.write("論文 PDF リンク")
            st.sidebar.write(selected_paper['linked_APA'], unsafe_allow_html=True)
            file_upload = st.sidebar.file_uploader("論文をアップロードしてください。", type=['pdf'])
            if file_upload:
                pdf_text = extract_text_without_headers_footers_from_stream(file_upload)
                st.session_state['chat_pdf_text'] = pdf_text
                print(f'pdf text length {len(pdf_text)}')
            else:
                st.sidebar.write("ファイルがアップロードされていません。")

        st.session_state['chat_is_open_access'] = is_open_access

def display_chat_option_component():
    if 'chat_pdf_text' in st.session_state:
        detail_button = st.sidebar.button("本文の要約の生成開始")
        if detail_button:
            paper_interpreter = gpt_japanese_paper_interpreter(st.session_state['chat_pdf_text'])
            # paper_interpreter = "これはテストです。"
            st.session_state['chat_interpreter'] = paper_interpreter
            st.sidebar.write("本文の要約を生成しました。")

def display_chat_component():
    if 'chat_selected_paper' in st.session_state and 'chat_pdf_text' in st.session_state:
        #論文全体の要約を表示
        if 'chat_interpreter' in st.session_state:
            st.sidebar.write(st.session_state['chat_interpreter'])

        #chat の履歴の表示
        for chat in st.session_state["chat_log"]:
            message = st.sidebar.chat_message(chat["role"])
            message.write(chat["content"])

        if 'chat_is_started' in st.session_state and st.session_state['chat_is_started']:
            return

        prompt = st.sidebar.text_input("論文について聞きたいことを入力してください。", value="")
        prompt_button = st.sidebar.button("送信")
        if prompt_button:
            if len(prompt) > 0:
                st.session_state['chat_is_started'] = True
                st.session_state["chat_log"].append({"role": "user", "content": prompt})
                st.session_state["chat_input"] = ""  # これでテキスト入力欄をリセット
                st.rerun()
            else:
                st.sidebar.write("メッセージが入力されていません。")
        # st.sidebar.write(st.session_state['chat_pdf_text'])

def chat_component():
    if 'chat_log' in st.session_state and 'chat_is_started' in st.session_state and st.session_state['chat_is_started']:
        st.session_state['chat_is_started'] = False
        # gpt_response = f"テスト中のためオウム返し: {prompt}"
        gpt_response = japanese_paper_chat(st.session_state['chat_pdf_text'], st.session_state['chat_log'])
        st.session_state["chat_log"].append({"role": "assistant", "content": gpt_response})
        st.rerun()

def chat_about_papers():
    # if not ('selected_number' in st.session_state and 'cluster_df_detail' in st.session_state):
    #     st.rerun()
    #サイドバーの幅の設定
    set_sidebar_width()

    if not ('selected_number' in st.session_state and 'cluster_df_detail' in st.session_state):
        st.stop()

    st.sidebar.title("論文の詳細情報")

    display_selected_paper_component()

    # 論文の基本情報を整理して表示する．
    display_paper_basic_information_component()

    display_spaces(2)

    display_japanese_abstract_component()

    display_spaces(2)

    display_open_access_paper_information_component()

    display_chat_option_component()

    display_spaces(2)

    display_chat_component()

    chat_component()