import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from utils import *
import time
import json
import streamlit as st
import requests
from collections import defaultdict, namedtuple
import datetime
import numpy as np
import japanize_matplotlib

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
            return ""

    # 'journal'列の各エントリにget_journal_name関数を適用して新しい列を作成します
    df['journal name'] = df['journal'].apply(get_journal_name)
    df['author names'] = df['authors'].apply(lambda x: [d.get('name') for d in x] if isinstance(x, list) else None)
    df['citation count'] = df['citationCount']
    df['published year'] = df['year'].apply(lambda x: str(x).replace('.0', ''))
    df = df[['Title', 'Importance', 'abstract', 'published year', 'citation count', 'journal name', 'author names']]
    df.columns = ['Title', 'Importance', 'Abstract', 'Published Year', 'Citation Count', 'Journal Name', 'Author Names']
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
    df['published year'] = df['year'].apply(lambda x: str(x).replace('.0', ''))
    df = df[['title', 'abstract', 'published year', 'citation count', 'journal name', 'author names']]
    df.columns =  ['Title',  'Abstract', 'Published Year', 'Citation Count', 'Journal Name', 'Author Names']
    st.dataframe(df.head(topk), hide_index=True)

def display_title():
    st.title("Enlighten Authoring")
    st.markdown("<strong>臨床的位置づけ立案支援AI: 専門情報のレビューと文章へのエビデンスの付与を行います</strong>", unsafe_allow_html=True)


def display_list(text_list, size=4):
    if not isinstance(text_list, list):
        st.write("文字列のリストを入力してください")
        return

    for text in text_list:
        st.write(
            f"<h{size} style='text-align: left;'> {text} </h{size}>",
            unsafe_allow_html=True,
        )


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
    x_ticks = range(int(paper_count_by_year.index.min()), current_year + 1)

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



def reset_session(session_state):
    keys_to_remove = ['papers', 'papers_df', 'H', 'cluster_candidates', 'cluster_df', 'selected_number',
                      'cluster_response', 'summary_response']
    for key in keys_to_remove:
        if key in session_state:
            session_state.pop(key)


def app():
    # refresh_button = st.button('Refresh button')
    # if refresh_button:
    #     st.experimental_rerun()

    debug_mode = st.checkbox("Debug Mode", value=True)
    if debug_mode:
        st.session_state['debug'] = True
    else:
        st.session_state['debug'] = False

    #Queryの管理
    display_title()

    if 'query' in st.session_state.keys():
        query = st.session_state['query']
    else:
        query = ""

    # Get the query from the user and sumit button
    query = st.text_input(
        "検索キーワードを入力",
        value = query,
        placeholder="e.g. \"pediatric epilepsy\" \"breast cancer\""
    )

    st.session_state['query'] = query

    #年の指定ボタン
    display_year_input()

    #検索の開始ボタン
    get_papers_button = st.button("Semantic Scholar で論文を検索")

    display_spaces(1)

    #更新ボタンが押されているときに再度検索するための設定
    if 'update_papers' in st.session_state and st.session_state['update_papers']:
        get_papers_button = True
        display_description("Semantic Scholar　で検索した論文を更新します")
        st.session_state.pop('update_papers')

    # 論文の取得, 結果の保存, 昔の検索結果の読み込み
    if query and get_papers_button:
        total_limit = 100
        with st.spinner("⏳ Semantic Scholar　から論文を取得しています..."):
            if os.path.exists(os.path.join(data_folder, f"{safe_filename(encode_to_filename(query))}.csv")) and st.session_state['debug']:
                display_description(f"{query} は Semantic Scholar で検索済みです。\n")
                papers_df = load_papers_dataframe(query)
                all_papers_df = load_papers_dataframe(query + '_all', [ 'authors'], ['title', 'abstract', 'year'])
                total = None
                st.session_state['update_papers'] = False
            else:
                display_description(f"{query} は Semantic Scholar で検索中です。\n")
                #Semantic Scholar による論文の保存
                #良い論文の100件の取得
                papers, total = get_papers(query, st.session_state['year'], total_limit=total_limit)
                # config への保存
                st.session_state['papers'] = papers
                if len(st.session_state['papers']) > 0:
                    papers_df = to_dataframe(st.session_state['papers'])
                    #all_papersへの入力はdataframe
                    all_papers_df = get_all_papers_from_references(papers_df)

        #検索結果がなかった場合に以前のデータフレームの削除．
        if 'papers' in st.session_state and len(st.session_state['papers']) == 0:
            display_description("Semantic Scholar での検索結果はありませんでした。")
            # reset_session(session_state=st.session_state)
            st.experimental_rerun()

        #csvの保存
        if not os.path.exists(os.path.join(data_folder, f"{safe_filename(encode_to_filename(query))}.csv")):
            save_papers_dataframe(papers_df, query)
            save_papers_dataframe(all_papers_df, query + '_all' )
            papers_df = load_papers_dataframe(query)
            all_papers_df = load_papers_dataframe(query + '_all', ['authors'], ['title', 'abstract', 'year'])

        st.session_state['all_papers_df'] = all_papers_df
        st.session_state['papers_df'] = papers_df

        display_spaces(2)

        #検索からの結果かデータベースに保存していた結果であることの表示
        if total:
            display_description(f"Semantic Scholar からの検索が完了しました。")
            display_description(f"{len(st.session_state['papers_df'])} / {total} の論文を取得しました。")
        else:
            display_description(f"データベースに保存されていた検索結果の読み込みが完了しました。")
            display_description(f"検索履歴から {len(st.session_state['papers_df'])} 件の論文を取得しました。")


    display_spaces(2)

    #すでに papers のデータフレームがあれば、それを表示する。
    if 'papers_df' in st.session_state:
        display_dataframe_detail(st.session_state['papers_df'],  f'論文検索結果上位 20 件', 20)

    display_spaces(2)

    # 論文の更新．ファイルの削除と再検索のための設定を行っている．
    if 'update_papers' in st.session_state:
        display_description("Semantic Scholar で検索した論文を更新します。")
        update_button = st.button("論文の更新")
        if update_button:
            csv_path = os.path.join(data_folder, f"{safe_filename(encode_to_filename(query))}.csv")
            all_csv_path = os.path.join(data_folder, f"{safe_filename(encode_to_filename(query))}_all.csv")
            if os.path.exists(csv_path):
                #保存してあるファイルの削除．
                os.remove(csv_path)
            if os.path.exists(all_csv_path):
                os.remove(all_csv_path)
            st.session_state['update_papers'] = True
            st.experimental_rerun()


    if 'papers_df' in st.session_state:
        display_spaces(2)
        display_description("AI レビュー生成", 3)

        st.session_state['number_of_review_papers'] = st.slider(
            f"レビューに使用する論文数を選択してください。",   min_value=1, value=20,  max_value=min(100, len(st.session_state['papers_df'])), step=1)

        toggle = display_language_toggle(f'レビュー生成')

        topk_review_button = st.button(f"上位 {st.session_state['number_of_review_papers']} 件の論文レビュー生成。(時間がかかります)",)
        if topk_review_button:
            with st.spinner("⏳ AIによるレビューの生成中です。 お待ち下さい..."):
                response, titles, caption = title_review_papers(st.session_state['papers_df'][:st.session_state['number_of_review_papers']], st.session_state['query'], model = 'gpt-4-32k', language=toggle)
                st.session_state['topk_review_caption'] = caption
                st.session_state['topk_review_response'] = response
                st.session_state['topk_review_titles'] = titles

                reference_indices = extract_reference_indices(response)
                references_list = [reference_text for i, reference_text in enumerate(titles) if i in reference_indices]
                st.session_state['topk_references_list'] = references_list

    #レビュー内容の常時表示
    if 'papers_df' in st.session_state and 'topk_review_response' in st.session_state and 'topk_review_caption' in st.session_state and 'topk_references_list' in st.session_state:
        display_description(st.session_state['topk_review_caption'], size=5)
        display_spaces(1)
        display_list(st.session_state['topk_review_response'].replace('#', '').split('\n'), size=8)

        if len(st.session_state['topk_references_list']) > 0:
            display_description('参考文献リスト', size=6)
            display_references_list(st.session_state['topk_references_list'])


    #レビューによる草稿の入力部分
    if 'papers_df' in st.session_state and 'topk_review_response' in st.session_state and 'topk_references_list' in st.session_state:
        display_spaces(1)
        display_description(f"文章の草稿を入力してください。上位 {st.session_state['number_of_review_papers']} 件のレビューによりエビデンスを付与します。", 5)
        #ドラフトの入力部分
        draft_text = st.text_area(label='review draft input filed.', placeholder='Past your draft of review here.', label_visibility='hidden', height=300)

        toggle = display_language_toggle(f"レビューによるエビデンス付与")

        write_summary_button = st.button(f"上位 {st.session_state['number_of_review_papers']} 件の論文レビューによるエビデンス付与。(時間がかかります)", )

        if write_summary_button and len(draft_text) > 0:
            with st.spinner("⏳ AIによるエビデンスの付与中です。 お待ち下さい..."):
                topk_summary_response, caption = summery_writer_with_draft(st.session_state['topk_review_response'], draft_text, st.session_state['topk_references_list'], model = 'gpt-4-32k', language=toggle)
                display_description(caption)
                display_spaces(2)

                st.session_state['topk_summary_response'] = topk_summary_response

                reference_indices = extract_reference_indices(topk_summary_response)
                references_list = [reference_text for i, reference_text in enumerate(st.session_state['topk_references_list']) if
                                   i in reference_indices]
                st.session_state['topk_summary_references_list'] = references_list
        elif write_summary_button:
            display_description("入力欄が空白です。草稿を入力してください。")

    #論文の草稿を書き直した結果の再表示
    if 'topk_summary_response' in st.session_state and 'topk_summary_references_list' in st.session_state:
        display_list(st.session_state['topk_summary_response'].replace('#', '').split('\n'), size=8)

        if len(st.session_state['topk_summary_references_list']) > 0:
            display_description('参考文献リスト', size=6)
            display_references_list(st.session_state['topk_summary_references_list'])

    #コミュニティグラフによるレビュー
    display_spaces(3)

    #サブグラフ(H)の構築
    if 'papers_df' in st.session_state:
        display_description('文献のクラスタリング', size=2)
        display_description("文献の引用関係に基づいてクラスタリングすることで、より関心に近いレビューを生成することができます")
        display_spaces(1)

        with st.spinner("⏳ コミュニティグラフの構築中です..."):
            G = get_paper_graph(st.session_state['papers_df'])
            st.session_state['G'] = G

        with st.spinner("⏳ サブグラフの構築中です..."):
            H = extract_subgraph(G)
            st.session_state['H'] = H

        node_attributes = pd.DataFrame.from_dict(dict(H.nodes(data=True)), orient='index')
        node_attributes.index.name = "Node Name (Paper ID)"


        if len(H.nodes) == 0:
            display_error("グラフの構築に失敗しました。申し訳ございませんが、Semantic Scholar の検索内容を変更して再検索してください。")
            st.session_state.pop('H')
            #グラフの構築に失敗した場合はここで停止する
            st.stop()

    if 'papers_df' in st.session_state and len(st.session_state['papers_df']) == 0:
        st.experimental_rerun()

    #cluster_df の構築
    if 'H' in st.session_state:
        with st.spinner(f"⏳ 論文のクラスタリング中です..."):
            cluster_counts, cluster_df, partition, clustering_result = community_clustering(st.session_state['H'])
            #ページランクによるソートアルゴリズム
            df_centrality = page_ranking_sort(st.session_state['H'])
            # すでに作成されている中心性のデータフレーム（df_centrality）に結合
            df_centrality['Cluster'] = df_centrality['Node'].map(clustering_result)
            #クラスターごとの keyword を作成
            cluster_keywords = calc_tf_idf(df_centrality)
            st.session_state['df_centrality'] = df_centrality

            st.session_state['cluster_keywords'] = cluster_keywords

            cluster_df = df_centrality.groupby('Cluster').agg({
                'Node': 'count',
                'DegreeCentrality': 'mean',
                'PageRank': 'mean'
            })
            cluster_df["ClusterKeywords"] = cluster_df.index.map(lambda x: ', '.join(cluster_keywords[x]))
            st.session_state['cluster_df'] = cluster_df

            #Cluster ID から Paper ID のリストを取得するリスト
            cluster_id_paper_ids = defaultdict(list)
            for key, value in partition.items():
                cluster_id_paper_ids[value].append(key)
            st.session_state['cluster_id_to_paper_ids'] = cluster_id_paper_ids

    if 'cluster_df' in st.session_state.keys():
        display_clusters = st.session_state['cluster_df'][st.session_state['cluster_df']['Node'] > 10]
        cluster_candidates = display_clusters.index.tolist()
        display_clusters = display_clusters.sort_values('Node', ascending=False)
        for cluster_number in display_clusters.index:
            selected_paper_ids = st.session_state['cluster_id_to_paper_ids'][cluster_number]
            extracted_df = st.session_state['papers_df'][st.session_state['papers_df']['paperId'].isin(selected_paper_ids)]
            display_clusters.loc[cluster_number, 'netNumberOfNodes'] = len(extracted_df)

        rename_columns = {
                'Node': "Number of Papers",
                "ClusterKeywords" : "Keywords"
            }
        display_clusters.rename(columns= rename_columns, inplace=True)

        display_dataframe(display_clusters, f'クラスタに含まれる文献数とキーワード', len(display_clusters), list(rename_columns.values()))

        st.session_state['cluster_candidates'] = cluster_candidates
        st.session_state['cluster_keywords'] = display_clusters['Keywords'].values


    # # 特定のクラスタについて,  nodeをすべて取り出す
    # # 取り出した node を持つ列について、
    #
    if 'cluster_candidates' in st.session_state and 'H' in st.session_state and 'G' in st.session_state and 'cluster_id_to_paper_ids' in st.session_state :
        assert len(st.session_state['cluster_candidates']) == len(st.session_state['cluster_keywords']), f"{len(st.session_state['cluster_candidates'])} : {len(st.session_state['cluster_keywords'])}"
        detailed_cluster_dict = {f'{cluster_number} : {cluster_keyword}' : cluster_number for cluster_number, cluster_keyword in zip(st.session_state['cluster_candidates'] , st.session_state['cluster_keywords'])}
        selected_number_key = st.selectbox('詳細を表示するクラスタ番号を選んでください。', detailed_cluster_dict.keys())
        display_spaces(2)
        selected_number = detailed_cluster_dict[selected_number_key]
        st.session_state['selected_number'] = selected_number

        cluster_df_detail = get_cluster_papers(st.session_state['G'], st.session_state['H'],
                                               st.session_state['cluster_id_to_paper_ids'][selected_number])
        st.session_state['cluster_df_detail'] = cluster_df_detail

        #ここで，all_papers から引っ張ってきた情報を表示させる．

        matched_papers_df = pd.merge(st.session_state['all_papers_df'], cluster_df_detail, left_on='paperId', right_on='Node', how='inner')
        # DegreeCentrality を結合する
        matched_papers_df['DegreeCentrality'] = matched_papers_df['Node'].map(
            cluster_df_detail.set_index('Node')['DegreeCentrality'])

        temp_cluster_df_detail =  matched_papers_df.rename(columns={'DegreeCentrality': "Importance"})
        temp_cluster_df_detail.index.name = 'Paper ID'
        temp_cluster_df_detail = temp_cluster_df_detail.sort_values('Importance', ascending=False)
        display_cluster_dataframe(temp_cluster_df_detail, f'クラスタ番号 {selected_number} 内での検索結果上位 20 件', 20)

        #クラスタの年情報の追加
        display_cluster_years(temp_cluster_df_detail)

        display_spaces(2)
        display_description(f'クラスタ番号{selected_number}の引用ネットワーク', size=3)
        with st.spinner(f"⏳ クラスタ番号 {selected_number} のグラフを描画中です..."):
            plot_cluster_i(st.session_state['H'], selected_number, st.session_state['df_centrality'])

    #特定のクラスタによる草稿の編集

    if 'selected_number' in st.session_state and 'cluster_df_detail' in st.session_state and 'query' in st.session_state:
        display_spaces(2)
        display_description(f'AI クラスタレビュー生成', size=2)

        cluster_df_detail = st.session_state['cluster_df_detail']

        st.session_state['number_of_cluster_review_papers'] = st.slider(
            f"クラスタレビューに使用する論文数を選択してください。",
                                                                    min_value=1,
                                                                      value=min(len(cluster_df_detail), 20),
                                                                      max_value=min(100, len(cluster_df_detail)), step=1)

        toggle = display_language_toggle(f"クラスタレビュー生成")

        #クラスタリングの結果のレビュー
        selected_review_button = st.button(f"クラスタ内上位{st.session_state['number_of_cluster_review_papers']}の論文レビュー生成。(時間がかかります)", )

        if selected_review_button:
            with st.spinner(f"⏳ AIによるクラスタレビューの生成中です。 お待ち下さい..."):
                selected_cluster_paper_ids = cluster_df_detail['Node'].values.tolist()[:st.session_state['number_of_cluster_review_papers']]
                result_list, result_dict = get_papers_from_ids(selected_cluster_paper_ids)
                selected_papers = pd.DataFrame(result_dict)
                cluster_response, reference_titles, caption = title_review_papers(selected_papers, st.session_state['query'], model = 'gpt-4-32k', language=toggle)

                display_description(caption)
                display_spaces(1)
                st.session_state['cluster_response'] = cluster_response
                st.session_state['cluster_reference_titles'] = reference_titles

            #response に含まれている Referenceの表示
                reference_indices = extract_reference_indices(cluster_response)
                references_list = [reference_text for i, reference_text in enumerate(reference_titles) if i in reference_indices]
                st.session_state['cluster_references_list'] = references_list



    if 'cluster_response' in st.session_state and 'cluster_references_list' in st.session_state and 'cluster_reference_titles' in st.session_state and 'selected_number' in st.session_state:
        display_description(st.session_state['cluster_response'])

        if len(st.session_state['cluster_references_list']) > 0:
            display_description('参考文献リスト', size=6)
            display_references_list(st.session_state['cluster_references_list'])

        #終了時にドラフトを入力できるようにする
        display_spaces(2)
        display_description(f"文章の草稿を入力してください。クラスタ内上位 {st.session_state['number_of_cluster_review_papers']} 件のレビューによりエビデンスを付与します。", 3)
        #ドラフトの入力部分
        draft_text = st.text_area(label='review draft input filed.', placeholder='Past your draft of review here!', label_visibility='hidden', height=300)

        toggle = display_language_toggle(f"クラスタレビューによるエビデンス付与")

        write_summary_button = st.button(f"クラスタ内上位{st.session_state['number_of_cluster_review_papers']}の論文レビューによるエビデンス付与。(時間がかかります)", )

        if write_summary_button and len(draft_text) > 0:
            with st.spinner("⏳ AIによるエビデンスの付与中です。 お待ち下さい..."):
                summary_response, caption = summery_writer_with_draft(st.session_state['cluster_response'], draft_text, st.session_state['cluster_references_list'], model = 'gpt-4-32k', language=toggle)
                display_description(caption)
                display_spaces(1)

                st.session_state['summary_response'] = summary_response

                reference_indices = extract_reference_indices(summary_response)
                references_list = [reference_text for i, reference_text in enumerate(st.session_state['cluster_reference_titles']) if
                                   i in reference_indices]
                st.session_state['summary_references_list'] = references_list
        elif write_summary_button:
            display_description("入力欄が空白です。草稿を入力してください。")

    #検索結果の再表示
    if 'summary_response' in st.session_state and 'summary_references_list' in st.session_state:
        display_list(st.session_state['summary_response'].replace('#', '').split('\n'), size=8)

        if len(st.session_state['summary_references_list']) > 0:
            display_description('参考文献リスト', size=6)
            display_references_list(st.session_state['summary_references_list'])


if __name__ == "__main__":
    app()
