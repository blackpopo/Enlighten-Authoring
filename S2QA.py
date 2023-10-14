import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from utils import *
import time
import json
import streamlit as st
import requests
from collections import defaultdict, namedtuple

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
DEEPL_API_KEY = st.secrets['DEEPL_API_KEY']
SEMANTICSCHOLAR_API_KEY = st.secrets['SEMANTICSCHOLAR_API_KEY']


def display_dataframe(df, title, topk, columns=None):
    st.subheader(title)
    if columns != None:
        df = df[columns]
    st.dataframe(df.head(topk))

def display_cluster_dataframe(df, title, topk):
    st.subheader(title)
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
    df['published year'] = df['year']
    df = df[['Title', 'Importance', 'abstract', 'published year', 'citation count', 'journal name', 'author names']]
    st.dataframe(df.head(topk), hide_index=True)

def display_dataframe_detail(df, title, topk):
    st.subheader(title)

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
    df['published year'] = df['year']
    df = df[['title', 'abstract', 'published year', 'citation count', 'journal name', 'author names']]
    st.dataframe(df.head(topk), hide_index=True)

def display_title():
    st.title("Enlighten Authoring")
    st.markdown("## _AI_ _Medical_ _Paper_ _Review_", unsafe_allow_html=True)


def display_list(text_list, size=4):
    if not isinstance(text_list, list):
        st.write("Please input the list of strings")
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


def generate_answer(prompt):
    """Generates an answer using ChatGPT."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to a researcher. You are helping them write a paper. You are given a prompt and a list of references. You are asked to write a summary of the references if they are related to the question. You should not include any personal opinions or interpretations in your answer, but rather focus on objectively presenting the information from the search results.",
            },
            {"role": "user", "content": prompt},
        ],
        api_key=OPENAI_API_KEY,
    )
    return response.choices[0].message.content

def display_spaces(repeat=1):
    for _ in range(repeat):
        st.markdown("<br>", unsafe_allow_html=True)


def display_references_list(references_list, size=7):
    with st.expander('Show references. Please click here!'):
        for reference_text in references_list:
            st.write(
                f"<h{size} style='text-align: left;'> {reference_text} </h{size}>",
                unsafe_allow_html=True,
            )

def display_language_toggle(unique_string):
    toggle = st.radio(
        f"Select the language for {unique_string}.",
        ['English', '日本語']
    )
    return toggle


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

def get_research_questions(answer):
    """Generates an answer using ChatGPT."""
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are helpful research visionary, consider the future possibilities and trends in a specific research area. Analyze the current state of the field, advancements in technology, and the potential for growth and development. Offer insights into how the researcher can contribute to this evolving landscape and provide innovative ideas that address challenges or gaps in the field. Inspire the researcher to think outside the box and explore new avenues of research, while also considering ethical, social, and environmental implications. Encourage collaboration and interdisciplinary approaches to maximize the impact of their work and drive the research area towards a promising and sustainable future.",
                },
                # {"role": "user", "content": answer },
                {"role": "user", "content": answer + "\n Instructions: Based on the literature review provided, please generate five detailed research questions for future researchers to explore. Your research questions should build upon the existing knowledge and address gaps or areas that require further investigation. Please provide sufficient context and details for each question."},
            ],
            api_key=OPENAI_API_KEY
            )
    return response.choices[0].message.content


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

    #Queryの管理
    display_title()

    if 'query' in st.session_state.keys():
        query = st.session_state['query']
    else:
        query = ""

    # Get the query from the user and sumit button
    query = st.text_input(
        "Enter your Research Keywords and Press Search Papers: ",
        value = query,
        placeholder="e.g. \"medical writer\" \"best assistant\""
    )

    st.session_state['query'] = query

    get_papers_button = st.button("Search Papers from Semantic Scholar")

    if 'update_papers' in st.session_state and st.session_state['update_papers']:
        get_papers_button = True
        display_description("Update Searched Papers from Semantic Scholar")
        st.session_state.pop('update_papers')

    display_spaces(1)

    # 論文の取得, 結果の保存, 昔の検索結果の読み込み
    if query and get_papers_button:
        total_limit = 100
        with st.spinner("⏳ Getting papers from semantic scholar..."):
            if os.path.exists(os.path.join(data_folder, f"{safe_filename(encode_to_filename(query))}.csv")):
                display_description(f"Query: </h5> <h2> {query} </h2> <h5> is already searched before.\n")
                papers_df = load_papers_dataframe(query)
                all_papers_df = load_papers_dataframe(query + '_all', [ 'authors'], ['title', 'abstract', 'year'])
                total = None
                st.session_state['update_papers'] = False
            else:
                display_description(f"Query: </h5> <h2> {query} </h2> <h5> is searching for semantic scholar.\n")
                #Semantic Scholar による論文の保存
                #良い論文の100件の取得
                papers, total = get_papers(query, total_limit=total_limit)
                # config への保存
                st.session_state['papers'] = papers
                if len(st.session_state['papers']) > 0:
                    papers_df = to_dataframe(st.session_state['papers'])
                    #all_papersへの入力はdataframe
                    all_papers_df = get_all_papers_from_references(papers_df)

        #検索結果がなかった場合に以前のデータフレームの削除．
        if 'papers' in st.session_state and len(st.session_state['papers']) == 0:
            display_description("No results found by Semantic Scholar")
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

        if total:
            display_description(f"Retrieval of papers from Semantic Scholar has been completed.")
            display_description(f"{len(st.session_state['papers_df'])} / {total} papers retrieved.")
        else:
            display_description(f"Retrieved from Semantic Scholar stored in database.")
            display_description(f"Up to {len(st.session_state['papers_df'])} papers are available for review.")

    if 'update_papers' in st.session_state:
        # 論文の更新．
        update_button = st.button("Update the papers")
        if update_button:
            #保存してあるファイルの削除．
            os.remove(os.path.join(data_folder, f"{safe_filename(encode_to_filename(query))}.csv"))
            os.remove(os.path.join(data_folder, f"{safe_filename(encode_to_filename(query))}_all.csv"))
            st.session_state['update_papers'] = True
            st.experimental_rerun()

    #すでに papers のデータフレームがあれば、それを表示する。
    if 'papers_df' in st.session_state:
        # if 'number_of_review_papers' in st.session_state:
        #     display_dataframe_detail(st.session_state['papers_df'], f'Top {st.session_state["number_of_review_papers"]} Papers retrieved from Semantic Scholar.', st.session_state["number_of_review_papers"])
        # else:
        display_dataframe_detail(st.session_state['papers_df'],  f'Top 20 Papers retrieved from Semantic Scholar.', 20)


    if 'papers_df' in st.session_state:
        display_spaces(2)
        display_description("Generate the Review of articles by AI based on Semantic Scholar search results.", 3)

        st.session_state['number_of_review_papers'] = st.slider(
            f"Number of papers that can be used for review: From 1 to {min(100, len(st.session_state['papers_df']))}.",
                                                                      min_value=1, value=20,  max_value=min(100, len(st.session_state['papers_df'])), step=1)

        toggle = display_language_toggle(f'top {st.session_state["number_of_review_papers"]} review')

        topk_review_button = st.button(f"Generate the review using the top {st.session_state['number_of_review_papers']} search results (Not necessary).",)
        if topk_review_button:
            with st.spinner("⏳ Currently working on the review using AI. Please wait..."):
                response, titles, caption = title_review_papers(st.session_state['papers_df'][:st.session_state['number_of_review_papers']], st.session_state['query'], model = 'gpt-4-32k', language=toggle)
                st.session_state['topk_review_caption'] = caption
                st.session_state['topk_review_response'] = response
                st.session_state['topk_review_titles'] = titles

    #レビュー内容の常時表示
    if 'papers_df' in st.session_state and 'topk_review_response' in st.session_state and 'topk_review_caption' in st.session_state:
        display_description(st.session_state['topk_review_caption'])
        display_spaces(1)
        display_description("Generated Review", size=3)
        display_list(st.session_state['topk_review_response'].replace('#', '').split('\n'), size=8)

    #コミュニティグラフによるレビュー
    display_spaces(3)

    #サブグラフ(H)の構築
    if 'papers_df' in st.session_state:
        display_description('Domain-specific review by community graph', size=2)
        display_spaces(1)

        with st.spinner("⏳ Graph is constructed using community graph ..."):
            G = get_paper_graph(st.session_state['papers_df'])
            st.session_state['G'] = G
            print('Graph is constructed')

        with st.spinner("⏳ Subgraph is constructed using community graph ..."):
            H = extract_subgraph(G)
            st.session_state['H'] = H
            print('Subgraph is constructed')

        node_attributes = pd.DataFrame.from_dict(dict(H.nodes(data=True)), orient='index')
        node_attributes.index.name = "Node Name (Paper ID)"


        if len(H.nodes) == 0:
            display_description("Graph construction failed, please try again from the Semantic Scholar search.")
            st.session_state.pop('H')

    if 'papers_df' in st.session_state and len(st.session_state['papers_df']) == 0:
        st.experimental_rerun()

    #cluster_df の構築
    if 'H' in st.session_state:
        with st.spinner(f"⏳ Papers are clustered now..."):
            cluster_counts, cluster_df, partition = community_clustering(st.session_state['H'])
            st.session_state['cluster_counts'] = cluster_counts
            st.session_state['cluster_df'] = cluster_df
            st.session_state['partition'] = partition
            cluster_id_paper_ids = defaultdict(list)
            for key, value in partition.items():
                cluster_id_paper_ids[value].append(key)
            st.session_state['cluster_id_to_paper_ids'] = cluster_id_paper_ids

    if 'cluster_df' in st.session_state.keys():
        display_clusters = st.session_state['cluster_df'][st.session_state['cluster_df']['numberOfNodes'] > 10]
        display_clusters = display_clusters.sort_values('density', ascending=False)
        cluster_candidates = display_clusters['clusterNumber'].values

        display_clusters.set_index('clusterNumber', inplace=True)
        for cluster_number in display_clusters.index:
            selected_paper_ids = st.session_state['cluster_id_to_paper_ids'][cluster_number]
            extracted_df = st.session_state['papers_df'][st.session_state['papers_df']['paperId'].isin(selected_paper_ids)]
            display_clusters.loc[cluster_number, 'netNumberOfNodes'] = len(extracted_df)

        rename_columns = {
                'clusterNumber': 'Cluster ID',
                'netNumberOfNodes': 'Number of Searched Papers in the Cluster',
                'numberOfNodes' : "Total Number of Papers in the Cluster",
                'density': "Importance"
            }
        display_clusters.rename(columns= rename_columns, inplace=True)

        display_dataframe(display_clusters, f'Cluster information constructed from search results about {st.session_state["query"]}', len(display_clusters), list(rename_columns.values())[1:])

        st.session_state['cluster_candidates'] = cluster_candidates

    # 特定のクラスターについて,  nodeをすべて取り出す
    # 取り出した node を持つ列について、

    if 'cluster_candidates' in st.session_state and 'H' in st.session_state and 'G' in st.session_state and 'cluster_id_to_paper_ids' in st.session_state and 'partition' in st.session_state:

        selected_number = st.selectbox('Please select the ID of cluster to get more information.', st.session_state['cluster_candidates'])
        display_spaces(2)
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
        display_cluster_dataframe(temp_cluster_df_detail, f'More information about the top 20 papers in Cluster ID {selected_number}.', 20)

        display_spaces(2)
        display_description(f'Cluster ID {selected_number} Community Subgraph', size=3)
        with st.spinner(f"⏳ Cluster ID {selected_number} Subgraph is plotting now..."):
            plot_cluster_i(st.session_state['H'], selected_number, st.session_state['partition'])

    #特定のクラスターによる草稿の編集

    if 'selected_number' in st.session_state and 'cluster_df_detail' in st.session_state and 'query' in st.session_state:
        display_spaces(2)
        display_description(f'Editing of drafts by AI and selected cluster ID {st.session_state["selected_number"]}', size=2)

        selected_number = st.session_state['selected_number']
        cluster_df_detail = st.session_state['cluster_df_detail']

        st.session_state['number_of_cluster_review_papers'] = st.slider(
            f"Number of papers in cluster id {selected_number} that can be used for editing: From 1 to {min(100, len(cluster_df_detail))}.",
                                                                    min_value=1,
                                                                      value=min(len(cluster_df_detail), 20),
                                                                      max_value=min(100, len(cluster_df_detail)), step=1)

        toggle = display_language_toggle(f"cluster {st.session_state['selected_number']} review")

        #クラスタリングの結果のレビュー
        selected_review_button = st.button(f"Generate the review using the top {st.session_state['number_of_cluster_review_papers']} papers. This may take some time.", )

        if selected_review_button:
            with st.spinner(f"⏳ Currently working on the review of Cluster ID {selected_number} using AI. Please wait..."):
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
            display_description('References', size=6)
            display_references_list(st.session_state['cluster_references_list'])

        #終了時にドラフトを入力できるようにする
        display_spaces(2)
        display_description('Please enter a draft of your paper in the input field', 3)
        #ドラフトの入力部分
        draft_text = st.text_area(label='review draft input filed.', placeholder='Past your draft of review here!', label_visibility='hidden', height=300)

        toggle = display_language_toggle(f"review draft")

        write_summary_button = st.button(f"Generate the re-edited version of your draft using the review of Cluster ID {st.session_state['selected_number']}. This may take some time.", )

        if write_summary_button and len(draft_text) > 0:
            with st.spinner("⏳ Generating the re-edited version of your draft with AI..."):
                summary_response, caption = summery_writer_with_draft(st.session_state['cluster_response'], draft_text, st.session_state['cluster_references_list'], model = 'gpt-4-32k', language=toggle)
                display_description(caption)
                display_spaces(1)

                st.session_state['summary_response'] = summary_response

                reference_indices = extract_reference_indices(summary_response)
                references_list = [reference_text for i, reference_text in enumerate(st.session_state['cluster_reference_titles']) if
                                   i in reference_indices]
                st.session_state['summary_references_list'] = references_list
        elif write_summary_button:
            display_description("The text input area is empty. Please fill in your draft.")

    #検索結果の再表示
    if 'summary_response' in st.session_state and 'summary_references_list' in st.session_state:
        display_list(st.session_state['summary_response'].replace('#', '').split('\n'), size=8)

        if len(st.session_state['summary_references_list']) > 0:
            display_description('References', size=6)
            display_references_list(st.session_state['summary_references_list'])



if __name__ == "__main__":
    app()
