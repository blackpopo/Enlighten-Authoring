import stat

import pandas as pd
import streamlit

from streamlit_utils import *
from utils import *

def set_sidebar_width():
    # サイドバーの幅設定
    st.markdown(
        """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 600px;
           max-width: 600px;
       }
       """,
        unsafe_allow_html=True,
    )
    pass

def get_journal_name(journal_info):
    try:
        if isinstance(journal_info, str):
            # journal_infoを辞書に変換し、'name'キーの値を取得します
            journal_info = ast.literal_eval(journal_info)
        return journal_info.get('name', '')
    except Exception as e:
        # 文字列が辞書として評価できない場合やキーが存在しない場合は空文字列を返します
        return "ジャーナルの情報がありません。"


def display_paper_basic_information(selected_paper_info):
    # 論文の基本情報を整理して表示
    if not selected_paper_info.empty:
        assert len(selected_paper_info) == 1
        paper = selected_paper_info.iloc[0]  # 最初の行を取得
        st.sidebar.write("論文タイトル:", paper['Title'])
        st.sidebar.write("著者:", paper['authors'][0]['name'] if len(paper['authors']) > 0 else "")
        with st.sidebar.expander("詳細情報", expanded=False):
            st.sidebar.write("出版年:", paper['published year'])
            st.sidebar.write("引用数:", str(paper['CitationCount']))
            st.sidebar.write("クラスタ:", paper['Cluster'])
            st.sidebar.write("アブストラクト:", paper['abstract'] if pd.notna(paper['abstract']) else "アブストラクトがありません。")
            st.sidebar.write("ジャーナル:", paper['journal name'])
            st.sidebar.write("オープンアクセス:", "はい" if paper['isOpenAccess'] else "いいえ")
        # 必要に応じて他の情報も表示
    else:
        st.sidebar.write("選択された論文の情報が見つかりません。")

def display_sidebar_component():
    if 'selected_number' in st.session_state and 'cluster_df_detail' in st.session_state:
        cluster_df_detail = st.session_state['cluster_df_detail']
        st.sidebar.title("論文の詳細情報")
        cluster_df_detail = cluster_df_detail.reset_index()
        cluster_df_detail['published year'] = cluster_df_detail['year'].apply(lambda x: str(x).replace('.0', ''))
        cluster_df_detail["first author"] = cluster_df_detail["authors"].apply(lambda x: x[0]['name'] if len(x) > 0 else "")
        cluster_df_detail["journal name"] = cluster_df_detail["journal"].apply(get_journal_name)
        # キー用の新しい列を作成
        cluster_df_detail['options'] = (cluster_df_detail.index + 1).astype(str) + " " + \
                                       cluster_df_detail['first author'].astype(str) + " (" + \
                                       cluster_df_detail['published year'].astype(str) + ") " + \
                                       cluster_df_detail['title'].astype(str)
        # 辞書を作成
        papers_dict = cluster_df_detail.copy().set_index('options')[['Node', 'isOpenAccess']].to_dict(orient='index')
        # 各行の(Node, isOpenAccess)をタプルとして格納
        papers_dict = {k: {"node": v['Node'], "isOpenAccess" : v['isOpenAccess']} for k, v in papers_dict.items()}

        selected_paper = st.sidebar.selectbox("詳細を表示したい論文を選択してください。", papers_dict.keys())

        st.session_state["chat_selected_paper"] = papers_dict[selected_paper]["node"]

        #論文の基本情報を整理して表示する．
        display_paper_basic_information(cluster_df_detail[cluster_df_detail["Node"] == st.session_state['chat_selected_paper']])

        #論文とのチャット部分
        japanese_abstract_button = st.sidebar.button("日本語のアブストラクトを表示しますか？")

        if japanese_abstract_button:
            st.sidebar.write("TODO")

        if papers_dict[selected_paper]["isOpenAccess"]:
            detail_button = st.sidebar.button("論文を取得して詳細を表示しますか？")
            if detail_button:
                st.sidebar.write("TODO")

def chat_about_papers():
    # if not ('selected_number' in st.session_state and 'cluster_df_detail' in st.session_state):
    #     st.rerun()
    #サイドバーの幅の設定
    set_sidebar_width()

    #サイドバーの表示
    display_sidebar_component()

