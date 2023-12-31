import clipboard
import streamlit.testing.element_tree

from streamlit_utils import *
from utils import *

def display_graph_component():
    display_description('文献のクラスタリング', size=2)
    display_description("文献の引用関係に基づいてクラスタリングすることで、より関心に近いレビューを生成することができます")
    display_spaces(1)

    citation_threshold = display_citation_threshold_toggle()
    if 'H' in st.session_state and 'citation_threshold' in st.session_state and st.session_state['citation_threshold'] != citation_threshold:
        st.session_state.pop('H')
    st.session_state['citation_threshold'] = citation_threshold

def construct_graph_component():
    #サブグラフ(H)の構築
    if 'papers_df' in st.session_state and 'G' not in st.session_state and 'H' not in st.session_state:
        with st.spinner("⏳ コミュニティグラフの構築中です..."):
            H, G = construct_direct_quotation_and_scrivener_combination(st.session_state['papers_df'], st.session_state['year'][3], st.session_state['citation_threshold'])
            st.session_state['G'] = G
            st.session_state['H'] = H

        node_attributes = pd.DataFrame.from_dict(dict(H.nodes(data=True)), orient='index')
        node_attributes.index.name = "Node Name (Paper ID)"
        if len(H.nodes) == 0:
            display_error("グラフの構築に失敗しました申し訳ございませんが、Semantic Scholar の検索内容を変更して再検索してください")
            st.session_state.pop('H')
            #グラフの構築に失敗した場合はここで停止する
            st.stop()


# _cluster_df DataFrameに新しい行を追加する関数
def add_all_row(_cluster_df):
    # 'Node'列の合計値を計算
    total_node = _cluster_df['Node'].sum()
    # 'Recent5YearsCount'列の合計値を計算
    total_recent5years_count = _cluster_df['Recent5YearsCount'].sum()
    # 'DegreeCentrality'列の平均値を計算
    mean_degree_centrality = _cluster_df['DegreeCentrality'].mean()
    # 'PageRank'列の平均値を計算
    mean_page_rank = _cluster_df['PageRank'].mean()
    # 'Year'列の平均値を計算
    mean_year = _cluster_df['Year'].mean()
    # 'CitationCount'列の合計値を計算
    citationCount = _cluster_df['CitationCount'].sum()
    # 新しい行を作成し、統計情報とClusterKeywords列を設定
    summary_row = pd.DataFrame({
        'Node': [total_node],
        'DegreeCentrality': [mean_degree_centrality],
        'PageRank': [mean_page_rank],
        'CitationCount': [citationCount],
        'Year': [mean_year],
        'Recent5YearsCount': [total_recent5years_count],
        'ClusterKeywords': ["All papers"]
    }).set_index(pd.Index(['all papers'], name='Cluster'))
    # _cluster_dfを逆順に並べ替え
    _cluster_df = _cluster_df.iloc[::-1]
    # 新しい行をDataFrameの先頭に追加
    _cluster_df = pd.concat([summary_row, _cluster_df])

    # 更新されたDataFrameを返す
    return _cluster_df


def setup_cluster_df(df_centrality):
    cluster_keywords = calc_tf_idf(df_centrality)
    cluster_df = df_centrality.groupby('Cluster').agg({
        'Node': 'count',
        'DegreeCentrality': 'mean',
        'PageRank': 'mean',
        'CitationCount': 'median',
        'Year': 'mean'
    })
    # 直近の年を特定
    # latest_year = df_centrality['Year'].max()
    latest_year = datetime.datetime.now().year
    # 直近5年間のデータを取得
    last_5_years = df_centrality[df_centrality['Year'] > latest_year - 5]
    # Clusterごとの件数を計算
    recent_counts = last_5_years.groupby('Cluster').size()
    # この件数をgbにマージ
    cluster_df['Recent5YearsCount'] = cluster_df.index.map(recent_counts)
    cluster_df['Recent5YearsCount'] = cluster_df['Recent5YearsCount'].fillna(0).astype(int)
    cluster_df["ClusterKeywords"] = cluster_df.index.map(lambda x: ', '.join(cluster_keywords[x]))
    # index 0 の追加
    cluster_df = add_all_row(cluster_df)
    cluster_df['Year'] = cluster_df['Year'].apply(lambda x: str(round(x, 2)))
    cluster_df['CitationCount'] = cluster_df['CitationCount'].apply(lambda x: str(round(x, 2)))
    return cluster_df, cluster_keywords


# @st.cache_data
def construct_cluster_component():
    if 'H' in st.session_state:
        with st.spinner(f"⏳ 論文のクラスタリング中です..."):
            #ページランクによる全体のソートアルゴリズム
            df_centrality = page_ranking_sort(st.session_state['H'], st.session_state['year'][1], st.session_state['year'][2], st.session_state['year'][3])
            #クラスターごとの keyword を作成
            st.session_state['partition'] = df_centrality.set_index('Node')['Cluster'].to_dict()
            cluster_id_paper_ids = df_centrality.groupby('Cluster')['Node'].apply(list).to_dict()

            cluster_df, cluster_keywords = setup_cluster_df(df_centrality)

            st.session_state['df_centrality'] = df_centrality
            st.session_state['cluster_keywords'] = cluster_keywords
            st.session_state['cluster_df'] = cluster_df
            st.session_state['cluster_id_to_paper_ids'] = cluster_id_paper_ids
            cluster_id_paper_ids["all papers"] = []
            #すべての paper_id を取得
            cluster_all_ids = []
            for ids in cluster_id_paper_ids.values():
                cluster_all_ids.extend(ids)

            cluster_id_paper_ids['all papers'] = sort_ids_by_reference_order(st.session_state['all_papers_df']['paperId'].tolist(), cluster_all_ids)
            st.session_state['cluster_id_to_paper_ids'] = cluster_id_paper_ids

def display_cluster_component():
    if 'cluster_df' in st.session_state:
        display_clusters = st.session_state['cluster_df'].copy()
        cluster_candidates = display_clusters.index.tolist()
        #あとで使用する情報の保存
        st.session_state['cluster_candidates'] = cluster_candidates
        st.session_state['cluster_keywords'] = display_clusters['ClusterKeywords'].values

        with st.spinner("⏳ クラスターの時間的な発展を描画中ですお待ち下さい"):
            _cluster_id_to_papers = st.session_state['cluster_id_to_paper_ids'].copy()
            del _cluster_id_to_papers['all papers']
            plot_research_front(st.session_state['df_centrality'],  st.session_state['H'],
                                st.session_state['cluster_df'].copy().drop("all papers"), _cluster_id_to_papers,
                                st.session_state['partition'])

def display_each_cluster_component():
    if 'cluster_candidates' in st.session_state and 'H' in st.session_state and 'G' in st.session_state and 'cluster_id_to_paper_ids' in st.session_state and len(st.session_state['cluster_candidates']) > 0:
        assert len(st.session_state['cluster_candidates']) == len(st.session_state['cluster_keywords']), f"{len(st.session_state['cluster_candidates'])} : {len(st.session_state['cluster_keywords'])}"
        detailed_cluster_dict = {f'{cluster_number} : {cluster_keyword}' : cluster_number for cluster_number, cluster_keyword in zip(st.session_state['cluster_candidates'] , st.session_state['cluster_keywords'])}
        selected_number_key = st.selectbox('詳細を表示するクラスタ番号を選んでください', detailed_cluster_dict.keys())
        display_spaces(2)
        #選択した番号で考える
        selected_number = detailed_cluster_dict[selected_number_key]
        st.session_state['selected_number'] = selected_number

        #ソートする順番の設定
        st.write("##### クラスター内の論文の並べ替え方を選択してください")
        cluster_sort = st.radio("クラスター内の論文の並べ替え方を選択してください", ['重要度', '出版年'], index=0, label_visibility='hidden')
        if 'cluster_sort' in st.session_state and st.session_state['cluster_sort'] != cluster_sort:
            st.session_state['cluster_sort'] = cluster_sort
            st.rerun()
        else:
            st.session_state['cluster_sort'] = cluster_sort

        display_spaces(1)


        #選択したクラスターで考える
        cluster_df_detail = get_cluster_papers(st.session_state['df_centrality'],  st.session_state['cluster_id_to_paper_ids'][selected_number])
        #ここで，all_papers から引っ張ってきた情報を表示させる．
        matched_papers_df = cluster_df_detail.merge(st.session_state['all_papers_df'], right_on='paperId', left_on='Node', how='inner')

        temp_cluster_df_detail =  matched_papers_df.rename(columns={'PageRank': "Importance"})
        temp_cluster_df_detail.index.name = 'Paper ID'

        if st.session_state['selected_number'] != 'all papers':
            if st.session_state['cluster_sort'] == "重要度":
                temp_cluster_df_detail = temp_cluster_df_detail.sort_values('Importance', ascending=False)
            elif  st.session_state['cluster_sort'] == "出版年":
                temp_cluster_df_detail = temp_cluster_df_detail.sort_values("year", ascending=False)
            else:
                raise ValueError(f"Invalid Sort { st.session_state['cluster_sort'] }")

        if 'number_of_cluster_review_papers' in st.session_state:
            display_cluster_dataframe(temp_cluster_df_detail, f'クラスタ番号 {selected_number} 内での検索結果上位 {st.session_state["number_of_cluster_review_papers"]} 件', st.session_state["number_of_cluster_review_papers"])
        else:
            display_cluster_dataframe(temp_cluster_df_detail, f'クラスタ番号 {selected_number} 内での検索結果上位 20 件', 20)

        # クラスターのレビュー生成
        number_of_papers = st.slider( f"クラスタレビューに使用する論文数を選択してください", min_value=1,
                value=min(len(cluster_df_detail), 20),  max_value=min(100, len(cluster_df_detail)), step=1)

        if 'number_of_cluster_review_papers' in st.session_state and st.session_state['number_of_cluster_review_papers'] != number_of_papers:
            st.session_state['number_of_cluster_review_papers'] = number_of_papers
            st.rerun()
        else:
            st.session_state['number_of_cluster_review_papers'] = number_of_papers

        # All year のみクラスタの年情報を表示する
        #クラスタの年情報の追加
        if st.session_state['selected_number'] == 'all papers':
            display_cluster_years(temp_cluster_df_detail)

        if st.session_state['selected_number'] != 'all papers':
            #streamlit にクラスタの情報を保存
            if st.session_state['cluster_sort'] == "重要度":
                matched_papers_df = matched_papers_df.sort_values('PageRank', ascending=False)
            elif  st.session_state['cluster_sort'] == "出版年":
                matched_papers_df = matched_papers_df.sort_values("year", ascending=False)
            else:
                raise ValueError(f"Invalid Sort { st.session_state['cluster_sort'] }")

        st.session_state['cluster_df_detail'] = matched_papers_df


def configure_cluster_review_component():
    if 'selected_number' in st.session_state and 'cluster_df_detail' in st.session_state and 'query' in st.session_state:
        display_spaces(2)
        display_description(f'AI クラスタレビュー生成', size=2)

        toggle = display_language_toggle(f"クラスタレビュー生成")
        st.session_state['cluster_review_toggle'] = toggle

        cluster_review_type = display_review_type_toggle("クラスタレビュー生成")
        st.session_state['cluster_review_type'] = cluster_review_type

        #クラスタリングの結果のレビュー
        selected_review_button = st.button(f"クラスタ内上位{st.session_state['number_of_cluster_review_papers']} 件の論文レビュー生成", )

        if selected_review_button:
            st.session_state['start_cluster_review'] = True
            st.session_state['cluster_start_index'] = 0
            st.session_state['cluster_review_response_list'] = []
            st.session_state['cluster_review_caption_list'] = []
            st.session_state['cluster_references_list_list'] = []
            st.session_state['cluster_draft_references_list_list'] = []


#実際のクラスタレビュー表示部分．
def display_cluster_review_component():
    if 'cluster_review_response_list' in st.session_state and 'cluster_references_list_list' in st.session_state  and 'cluster_review_caption_list' in st.session_state and 'cluster_draft_references_list' in st.session_state:
        assert len(st.session_state['cluster_review_response_list']) == len(st.session_state['cluster_review_caption_list'])\
                                                                == len(st.session_state['cluster_references_list_list']) \
                                                                == len(st.session_state['cluster_draft_references_list_list'])

        for index, cluster_review_caption, cluster_review_response, cluster_references_list in zip(range(1, len(st.session_state['cluster_review_caption_list']) + 1), st.session_state['cluster_review_caption_list'],
                                                                                            st.session_state['cluster_review_response_list'], st.session_state['cluster_references_list_list']):
            display_description(cluster_review_caption, size=5)
            display_spaces(1)
            st.markdown(cluster_review_response, unsafe_allow_html=True)

            if len(cluster_references_list) > 0:
                display_description('参考文献リスト', size=6)
                display_references_list(cluster_references_list)

            copy_button = st.button(f"{index}番目のレビューコピー")
            if copy_button:
                clipboard.copy(cluster_review_response)
                st.write("レビューをコピーしました。")
            display_spaces(2)

def display_next_cluster_review_component():
    if 'cluster_review_response_list' in st.session_state and 'cluster_references_list_list' in st.session_state  and 'cluster_review_caption_list' in st.session_state and 'cluster_draft_references_list' in st.session_state:
        next_cluster_review_button = st.button(
            f"次の上位 {st.session_state['number_of_cluster_review_papers']} 件の論文によるクラスタレビュー生成")
        if next_cluster_review_button:
            st.session_state['cluster_start_index'] += 1
            st.session_state['start_cluster_review'] = True

def generate_cluster_review_component():
    if 'start_cluster_review' in st.session_state and st.session_state['start_cluster_review']:
        cluster_df_detail = st.session_state['cluster_df_detail']
        st.session_state['start_cluster_review'] = False
        start_index = st.session_state['number_of_cluster_review_papers'] * st.session_state['cluster_start_index']
        end_index = st.session_state['number_of_cluster_review_papers'] * (st.session_state['cluster_start_index'] + 1)
        last_index = min(end_index, len(cluster_df_detail))
        if start_index >= len(cluster_df_detail):
            st.write("### 当該クラスタ内でのAI によるクラスタレビューはこれ以上生成できません．")
            return

        with st.spinner(f"⏳ AIによるクラスタレビューの生成中です お待ち下さい..."):
            selected_cluster_paper_ids = cluster_df_detail['Node'].values.tolist()[start_index: end_index]
            result_list, result_dict = get_papers_from_ids(selected_cluster_paper_ids)
            selected_papers = set_paper_information(pd.DataFrame(result_dict))

            if st.session_state['cluster_review_type'] == '通常':
                cluster_response, reference_links, caption, draft_references = streamlit_title_review_papers(
                    selected_papers, st.session_state['query'], model='gpt-4-1106-preview',
                    language=st.session_state['cluster_review_toggle'])
            elif st.session_state['cluster_review_type'] == '長文':
                cluster_response, reference_links, caption, draft_references = streamlit_title_long_review_papers(
                    selected_papers, st.session_state['query'], model='gpt-4-1106-preview',
                    language=st.session_state['cluster_review_toggle'])
            else:
                raise ValueError(f"Invalid Review Type {st.session_state['cluster_review_type']}")
            display_description(caption)
            display_spaces(1)
            st.session_state['cluster_review_response'] = cluster_response
            st.session_state['cluster_review_caption'] = f"クラスタ内上位{start_index + 1} ~ {last_index}件"

            st.session_state['cluster_review_response_list'].append(cluster_response)
            st.session_state['cluster_review_caption_list'].append(f"クラスタ内上位{start_index + 1} ~ {last_index}件")

            # response に含まれている Referenceの表示
            reference_indices = extract_reference_indices(cluster_response)
            references_list = [reference_text for i, reference_text in enumerate(reference_links) if
                               i in reference_indices]
            draft_references_list = [reference_text for i, reference_text in enumerate(draft_references) if
                                     i in reference_indices]
            st.session_state['cluster_references_list'] = references_list
            st.session_state['cluster_draft_references_list'] = draft_references_list

            st.session_state['cluster_references_list_list'].append(references_list)
            st.session_state['cluster_draft_references_list_list'].append(draft_references_list)
            st.rerun()

def generate_cluster_draft_component():
    if 'number_of_cluster_review_papers' in st.session_state:
        display_description(f"文章の草稿を入力してくださいクラスタ内上位 {st.session_state['number_of_cluster_review_papers']} 件のレビューによりエビデンスを付与します", 3)
        #ドラフトの入力部分
        if not 'cluster_review_draft_text' in st.session_state:
            draft_text = st.text_area(label='cluster review draft input filed.', placeholder='Past your draft of review here.', label_visibility='hidden', height=300)
        else:
            draft_text = st.text_area(label='cluster review draft input filed.', value = st.session_state['cluster_review_draft_text'], placeholder='Past your draft of review here.', label_visibility='collapsed', height=300)
        st.session_state['cluster_review_draft_text'] = draft_text

        toggle = display_language_toggle(f"クラスタレビューによるエビデンス付与")
        mode_toggle = display_draft_evidence_toggle(f"クラスタレビュー生成")

        write_summary_button = st.button(f"クラスタ内上位{st.session_state['number_of_cluster_review_papers']}の論文レビューによるエビデンス付与", )

        if write_summary_button and len(draft_text) > 0:
            if 'cluster_review_response' in st.session_state and len(st.session_state['cluster_review_response']) > 0 and 'cluster_draft_references_list' in st.session_state:
                with st.spinner("⏳ AIによるエビデンスの付与中です お待ち下さい..."):
                    summary_response, caption = streamlit_summery_writer_with_draft(st.session_state['cluster_review_response'], draft_text, st.session_state['cluster_draft_references_list'], model = 'gpt-4-1106-preview', language=toggle, mode=mode_toggle)
                    display_description(caption)
                    display_spaces(1)

                    st.session_state['summary_response'] = summary_response

                    reference_indices = extract_reference_indices(summary_response)
                    references_list = [reference_text for i, reference_text in enumerate(st.session_state['cluster_references_list']) if
                                       i in reference_indices]
                    st.session_state['summary_references_list'] = references_list
            else:
                display_description("クラスタレビューがありません先にクラスタレビューを生成してください")
        elif write_summary_button:
            display_description("入力欄が空白です草稿を入力してください")

def display_cluster_draft_component():
    if 'summary_response' in st.session_state and 'summary_references_list' in st.session_state:
        # display_list(st.session_state['summary_response'].replace('#', '').split('\n'), size=8)
        st.markdown(st.session_state['summary_response'], unsafe_allow_html=True)

        if len(st.session_state['summary_references_list']) > 0:
            display_description('参考文献リスト', size=6)
            display_references_list(st.session_state['summary_references_list'])

        copy_button = st.button(f"レビューコピー")
        if copy_button:
            clipboard.copy(st.session_state['summary_response'])
            st.write("レビューをコピーしました。")
        display_spaces(2)

def cluster_review_papers():
    #papers_df がない場合にはやり直す
    if 'papers_df' in st.session_state and len(st.session_state['papers_df']) == 0:
        st.rerun()

    display_graph_component()

    #cluster_df の構築
    construct_graph_component()

    construct_cluster_component()

    display_spaces(1)

    #クラスターの詳細表示
    display_cluster_component()

    display_each_cluster_component()

    display_spaces(2)

    #クラスターのレビュー生成

    configure_cluster_review_component()

    display_cluster_review_component()

    display_next_cluster_review_component()

    generate_cluster_review_component()

    display_spaces(2)

    #クラスターの草稿編集

    generate_cluster_draft_component()

    display_cluster_draft_component()

