from streamlit_utils import *
from utils import *

def construct_graph_component():
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

def construct_cluster_component():
    if 'H' in st.session_state:
        with st.spinner(f"⏳ 論文のクラスタリング中です..."):
            cluster_counts, partition, clustering_result = community_clustering(st.session_state['H'])
            #ページランクによる全体のソートアルゴリズム
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
                    'PageRank': 'mean',
                    'CitationCount': 'median',
                    'Year': 'mean'
                })
            cluster_df['Year'] = cluster_df['Year'].apply(lambda x: str(round(x, 2)))

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

            st.session_state['cluster_df'] = cluster_df
            st.session_state['partition'] = partition

            #Cluster ID から Paper ID のリストを取得するリスト
            cluster_id_paper_ids = defaultdict(list)
            for key, value in partition.items():
                cluster_id_paper_ids[value].append(key)
            st.session_state['cluster_id_to_paper_ids'] = cluster_id_paper_ids

def display_cluster_component():
    if 'cluster_df' in st.session_state.keys():
        display_clusters = st.session_state['cluster_df'][st.session_state['cluster_df']['Node'] > 10]
        cluster_candidates = display_clusters.index.tolist()
        display_clusters = display_clusters.sort_values('Node', ascending=False)
        #あとで使用する情報の保存
        st.session_state['cluster_candidates'] = cluster_candidates
        st.session_state['cluster_keywords'] = display_clusters['ClusterKeywords'].values

        for cluster_number in display_clusters.index:
            selected_paper_ids = st.session_state['cluster_id_to_paper_ids'][cluster_number]
            extracted_df = st.session_state['papers_df'][st.session_state['papers_df']['paperId'].isin(selected_paper_ids)]
            display_clusters.loc[cluster_number, 'netNumberOfNodes'] = len(extracted_df)

        rename_columns = {
                'Node': "文献数",
                "Year": "平均年",
                'Recent5YearsCount' : "直近5年の文献数",
                "ClusterKeywords": "キーワード",
            }
        display_clusters.index.name = "クラスタ番号"
        display_clusters.rename(columns= rename_columns, inplace=True)

        display_dataframe(display_clusters, f'クラスタに含まれる文献数とキーワード', len(display_clusters), list(rename_columns.values()))

def display_each_cluster_component():
    if 'cluster_candidates' in st.session_state and 'H' in st.session_state and 'G' in st.session_state and 'cluster_id_to_paper_ids' in st.session_state :
        assert len(st.session_state['cluster_candidates']) == len(st.session_state['cluster_keywords']), f"{len(st.session_state['cluster_candidates'])} : {len(st.session_state['cluster_keywords'])}"
        detailed_cluster_dict = {f'{cluster_number} : {cluster_keyword}' : cluster_number for cluster_number, cluster_keyword in zip(st.session_state['cluster_candidates'] , st.session_state['cluster_keywords'])}
        selected_number_key = st.selectbox('詳細を表示するクラスタ番号を選んでください。', detailed_cluster_dict.keys())
        display_spaces(2)
        cluster_sort = st.radio("クラスター内の論文の並べ替え方を選択してください。", ['重要度', '出版年'], index=0)
        display_spaces(1)
        selected_number = detailed_cluster_dict[selected_number_key]
        st.session_state['selected_number'] = selected_number

        #選択したクラスターで考える
        cluster_df_detail = get_cluster_papers(st.session_state['G'], st.session_state['H'],
                                               st.session_state['cluster_id_to_paper_ids'][selected_number])
        st.session_state['cluster_df_detail'] = cluster_df_detail

        #ここで，all_papers から引っ張ってきた情報を表示させる．
        matched_papers_df = cluster_df_detail.merge(st.session_state['all_papers_df'], right_on='paperId', left_on='Node', how='inner')
        # DegreeCentrality を結合する
        # matched_papers_df['PageRank'] = matched_papers_df['PageRank'].map(
        #     cluster_df_detail.set_index('Node')['PageRank'])
        temp_cluster_df_detail =  matched_papers_df.rename(columns={'PageRank': "Importance"})
        temp_cluster_df_detail.index.name = 'Paper ID'

        if cluster_sort == "重要度":
            temp_cluster_df_detail = temp_cluster_df_detail.sort_values('Importance', ascending=False)
        elif cluster_sort == "出版年":
            temp_cluster_df_detail = temp_cluster_df_detail.sort_values("year", ascending=False)
        else:
            raise ValueError(f"Invalid Sort {cluster_sort}")
        display_cluster_dataframe(temp_cluster_df_detail, f'クラスタ番号 {selected_number} 内での検索結果上位 20 件', 20)

        #クラスタの年情報の追加
        display_cluster_years(temp_cluster_df_detail)


        display_spaces(2)
        display_description(f'クラスタ番号{selected_number}の引用ネットワーク', size=5)
        with st.expander(label='引用ネットワーク'):
            with st.spinner(f"⏳ クラスタ番号 {selected_number} のグラフを描画中です..."):
                plot_cluster_i(st.session_state['H'], selected_number, st.session_state['partition'])

def generate_cluster_review_component():
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
        st.session_state['cluster_review_toggle'] = toggle

        #クラスタリングの結果のレビュー
        selected_review_button = st.button(f"クラスタ内上位{st.session_state['number_of_cluster_review_papers']}の論文レビュー生成。(時間がかかります)", )

        if selected_review_button:
            with st.spinner(f"⏳ AIによるクラスタレビューの生成中です。 お待ち下さい..."):
                selected_cluster_paper_ids = cluster_df_detail['Node'].values.tolist()[:st.session_state['number_of_cluster_review_papers']]
                result_list, result_dict = get_papers_from_ids(selected_cluster_paper_ids)
                selected_papers = set_paper_information(pd.DataFrame(result_dict))
                cluster_response, reference_links, caption, draft_references = title_review_papers(selected_papers, st.session_state['query'], model = 'gpt-4-32k', language=toggle)

                display_description(caption)
                display_spaces(1)
                st.session_state['cluster_review_response'] = cluster_response
                st.session_state['cluster_review_caption'] = f"クラスタ内上位{caption}"

            #response に含まれている Referenceの表示
                reference_indices = extract_reference_indices(cluster_response)
                references_list = [reference_text for i, reference_text in enumerate(reference_links) if i in reference_indices]
                draft_references_list = [reference_text for i, reference_text in enumerate(draft_references) if i in reference_indices]
                st.session_state['cluster_references_list'] = references_list
                st.session_state['cluster_draft_references_list'] = draft_references_list

def display_cluster_review_component():
    if 'cluster_review_response' in st.session_state and 'cluster_references_list' in st.session_state  and 'selected_number' in st.session_state:
        display_description(st.session_state['cluster_review_caption'], size=5)
        display_spaces(1)
        display_description(st.session_state['cluster_review_response'])

        if len(st.session_state['cluster_references_list']) > 0:
            display_description('参考文献リスト', size=6)
            display_references_list(st.session_state['cluster_references_list'])

def generate_next_cluster_review_component():
    if 'cluster_df_detail' in st.session_state and 'cluster_review_response' in st.session_state  and 'cluster_references_list' in st.session_state and 'number_of_cluster_review_papers' in st.session_state:
        next_cluster_review_button = st.button(f"次の上位 {st.session_state['number_of_cluster_review_papers']} 件の論文によるクラスタレビュー生成。(時間がかかります)")
        if next_cluster_review_button:
            if not 'next_number_of_cluster_review_papers' in st.session_state:
                st.session_state['next_number_of_cluster_review_papers'] = st.session_state['number_of_cluster_review_papers'] * 2
            elif ('next_number_of_cluster_review_papers' in st.session_state) and (
                    st.session_state['next_number_of_cluster_review_papers'] < len(st.session_state['cluster_df_detail'])):
                st.session_state['next_number_of_cluster_review_papers'] = st.session_state['number_of_cluster_review_papers'] + \
                                                                   st.session_state['next_number_of_cluster_review_papers']
            else:
                st.session_state['next_number_of_cluster_review_papers'] = st.session_state['number_of_cluster_review_papers']

            button_title = f"クラスタ内上位 {st.session_state['next_number_of_cluster_review_papers'] - st.session_state['number_of_cluster_review_papers'] + 1} 件目から {min(st.session_state['next_number_of_cluster_review_papers'], len(st.session_state['cluster_df_detail']))} 件目"

            with st.spinner(f"⏳ {button_title} の論文を使用した AI によるクラスタレビューの生成中です。 お待ち下さい..."):
                selected_cluster_paper_ids = st.session_state['cluster_df_detail']['Node'].values.tolist()[st.session_state['next_number_of_cluster_review_papers'] - st.session_state['number_of_cluster_review_papers']:st.session_state['next_number_of_cluster_review_papers']]
                result_list, result_dict = get_papers_from_ids(selected_cluster_paper_ids)
                selected_papers = set_paper_information(pd.DataFrame(result_dict))

                response, links, caption, draft_references = title_review_papers(selected_papers,
                    st.session_state['query'], model='gpt-4-32k', language=st.session_state['cluster_review_toggle'] )
                st.session_state['cluster_review_response'] = response
                st.session_state['cluster_review_caption'] = f"{button_title}の論文による" + caption

                reference_indices = extract_reference_indices(response)
                references_list = [reference_text for i, reference_text in enumerate(links) if i in reference_indices]
                draft_references_list = [reference_text for i, reference_text in enumerate(draft_references) if i in reference_indices]
                st.session_state['cluster_references_list'] = references_list
                st.session_state['cluster_draft_references_list'] = draft_references_list
                st.experimental_rerun()

def generate_cluster_draft_component():
    if 'number_of_cluster_review_papers' in st.session_state:
        display_description(f"文章の草稿を入力してください。クラスタ内上位 {st.session_state['number_of_cluster_review_papers']} 件のレビューによりエビデンスを付与します。", 3)
        #ドラフトの入力部分
        if not 'cluster_review_draft_text' in st.session_state:
            draft_text = st.text_area(label='cluster review draft input filed.', placeholder='Past your draft of review here.', label_visibility='hidden', height=300)
        else:
            draft_text = st.text_area(label='clsuter review draft input filed.', value = st.session_state['cluster_review_draft_text'],placeholder='Past your draft of review here.', label_visibility='hidden', height=300)
        st.session_state['cluster_review_draft_text'] = draft_text

        toggle = display_language_toggle(f"クラスタレビューによるエビデンス付与")
        mode_toggle = display_draft_evidence_toggle(f"クラスタレビュー生成")

        write_summary_button = st.button(f"クラスタ内上位{st.session_state['number_of_cluster_review_papers']}の論文レビューによるエビデンス付与。(時間がかかります)", )

        if write_summary_button and len(draft_text) > 0:
            if 'cluster_review_response' in st.session_state and 'cluster_draft_references_list' in st.session_state:
                with st.spinner("⏳ AIによるエビデンスの付与中です。 お待ち下さい..."):
                    summary_response, caption = summery_writer_with_draft(st.session_state['cluster_review_response'], draft_text, st.session_state['cluster_draft_references_list'], model = 'gpt-4-32k', language=toggle, mode=mode_toggle)
                    display_description(caption)
                    display_spaces(1)

                    st.session_state['summary_response'] = summary_response

                    reference_indices = extract_reference_indices(summary_response)
                    references_list = [reference_text for i, reference_text in enumerate(st.session_state['cluster_references_list']) if
                                       i in reference_indices]
                    st.session_state['summary_references_list'] = references_list
            else:
                display_description("クラスタレビューがありません。先にクラスタレビューを生成してください。")
        elif write_summary_button:
            display_description("入力欄が空白です。草稿を入力してください。")

def display_cluster_draft_component():
    if 'summary_response' in st.session_state and 'summary_references_list' in st.session_state:
        display_list(st.session_state['summary_response'].replace('#', '').split('\n'), size=8)

        if len(st.session_state['summary_references_list']) > 0:
            display_description('参考文献リスト', size=6)
            display_references_list(st.session_state['summary_references_list'])

def cluster_review_papers():
    construct_cluster_component()

    #papers_df がない場合にはやり直す
    if 'papers_df' in st.session_state and len(st.session_state['papers_df']) == 0:
        st.experimental_rerun()

    #cluster_df の構築
    construct_graph_component()

    construct_cluster_component()

    display_spaces(1)

    #クラスターの詳細表示
    display_cluster_component()

    display_each_cluster_component()

    display_spaces(2)

    #クラスターのレビュー生成

    generate_cluster_review_component()

    display_cluster_review_component()

    generate_next_cluster_review_component()

    display_spaces(2)

    #クラスターの草稿編集

    generate_cluster_draft_component()

    display_cluster_draft_component()

