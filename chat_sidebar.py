import pandas as pd

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

def display_selected_paper_component():
    cluster_df_detail = st.session_state['cluster_df_detail'].copy()
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
    papers_dict = {k: {"node": v['Node'], "isOpenAccess": v['isOpenAccess']} for k, v in papers_dict.items()}

    selected_paper_node = st.sidebar.selectbox("詳細を表示したい論文を選択してください。", papers_dict.keys())

    if not 'chat_selected_paper_node' in st.session_state or st.session_state['chat_selected_paper_node'] != \
            papers_dict[selected_paper_node]["node"]:
        for key in st.session_state:
            if 'chat_' in key:
                st.session_state.pop(key)
        st.session_state["chat_selected_paper_node"] = papers_dict[selected_paper_node]["node"]
        st.session_state["chat_log"] = [{"role": "assistant" , "content": "論文について聞きたいことを入力してください。"}]

    selected_paper = cluster_df_detail[cluster_df_detail["Node"] == st.session_state['chat_selected_paper_node']]
    st.session_state['chat_selected_paper'] = selected_paper.iloc[0]

def display_paper_basic_information_component():
    if 'chat_selected_paper' in st.session_state:
        paper = st.session_state['chat_selected_paper']
        # 論文の基本情報を整理して表示
        if not paper.empty:
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


def display_open_access_paper_information_component():
    if 'chat_selected_paper' in st.session_state and not 'chat_pdf_text' in st.session_state:
        selected_paper = st.session_state['chat_selected_paper']
        if selected_paper['isOpenAccess'] and pd.notna(selected_paper["openAccessPdf"]):
            detail_button = st.sidebar.button("チャットを開始")
            if detail_button:
                pdf_url = selected_paper["openAccessPdf"]["url"]
                pdf_path = download_pdf(pdf_url)
                if not pdf_path:
                    st.sidebar.write("Open Access の pdf が取得できませんでした。ファイルをアップロードしてください。")
                    st.sidebar.write(selected_paper['linked_APA'], unsafe_allow_html=True)
                    file_upload = st.sidebar.file_uploader("論文をアップロードしてください。", type=['pdf'])
                    if  file_upload:
                        pdf_text = extract_text_without_headers_footers_from_stream(file_upload)
                        st.session_state['chat_pdf_text'] = pdf_text
                    else:
                        st.sidebar.write("ファイルがアップロードされていません。")
                else:
                    pdf_text = extract_text_without_headers_footers(pdf_path)
                    st.session_state['chat_pdf_text'] = pdf_text

        else:
            st.sidebar.write("論文の PDF のリンク")
            st.sidebar.write(selected_paper['linked_APA'], unsafe_allow_html=True)
            file_upload = st.sidebar.file_uploader("論文をアップロードしてください。", type=['pdf'])
            detail_button = st.sidebar.button("チャットを開始")
            if detail_button:
                if file_upload:
                    pdf_text = extract_text_without_headers_footers_from_stream(file_upload)
                    st.session_state['chat_pdf_text'] = pdf_text
                else:
                    st.sidebar.write("ファイルがアップロードされていません。")

def display_chat_component():
    if 'chat_selected_paper' in st.session_state and 'chat_pdf_text' in st.session_state:
        #chat の履歴の表示
        for chat in st.session_state["chat_log"]:
            message = st.sidebar.chat_message(chat["role"])
            message.write(chat["content"])

        #chat の入力部分
        prompt = st.sidebar.text_input("論文について聞きたいことを入力してください。")
        prompt_button = st.sidebar.button("送信")
        if prompt_button:
            if len(prompt) > 0:
                st.sidebar.write(f"User has sent the following prompt: {prompt}")
                st.session_state["chat_log"].append({"role": "user", "content" : prompt})
                st.sidebar.write(f"ユーザーの入力内容 {prompt}")
                gpt_response = prompt
                st.session_state["chat_log"].append({"role": "assistant", "content": gpt_response})
                st.rerun()
            else:
                st.sidebar.write("メッセージが入力されていません。")
        # st.sidebar.write(st.session_state['chat_pdf_text'])



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

    display_spaces(2)

    display_chat_component()