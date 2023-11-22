import streamlit.runtime.state

from streamlit_utils import *
import os
from utils import *

def research_papers():
    #検索の開始ボタン
    get_papers_button = st.button("Semantic Scholar で論文を検索")

    query = st.session_state['query']
    display_spaces(1)
    # 論文の取得, 結果の保存, 昔の検索結果の読み込み
    if len(query)  > 0  and get_papers_button:
        total_limit = 100
        all_reset_session(session_state=st.session_state, except_key=['debug', 'query', 'year'])
        with st.spinner("⏳ Semantic Scholar　から論文を取得しています..."):
            if os.path.exists(os.path.join(data_folder, f"{safe_filename(encode_to_filename(query))}.csv")) and st.session_state['debug']:
                display_description(f"{query} を Semantic Scholar で検索済みです。\n")
                papers_df = load_papers_dataframe(query)
                all_papers_df = load_papers_dataframe(query + '_all', [ 'authors', 'citationStyles'])
                total = None
            else:
                display_description(f"{query} を Semantic Scholar で検索中です。\n")
                #Semantic Scholar による論文の保存
                #良い論文の100件の取得
                papers, total = get_papers(query, st.session_state['year'], limit=20, total_limit=total_limit)
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
        if not os.path.exists(os.path.join(data_folder, f"{safe_filename(encode_to_filename(query))}.csv")) and st.session_state['debug']:
            save_papers_dataframe(papers_df, query)
            save_papers_dataframe(all_papers_df, query + '_all' )
            papers_df = load_papers_dataframe(query)
            all_papers_df = load_papers_dataframe(query + '_all', ['authors', 'citationStyles'])

        print(f"all papers df size {len(all_papers_df)} papers df size {len(papers_df)}")
        st.session_state['all_papers_df'] = set_paper_information(all_papers_df)
        st.session_state['papers_df'] = set_paper_information(papers_df)

        display_spaces(2)

        #検索からの結果かデータベースに保存していた結果であることの表示
        if total:
            display_description(f"Semantic Scholar からの検索が完了しました。")
            display_description(f"{len(st.session_state['papers_df'])} / {total} の論文を取得しました。\n参考文献を含めて {len(st.session_state['all_papers_df'])} 件の論文を取得しました。")
        else:
            display_description(f"データベースに保存されていた検索結果の読み込みが完了しました。")
            display_description(f"検索履歴から {len(st.session_state['papers_df'])} 件の論文を取得しました。\n参考文献を含めて {len(st.session_state['all_papers_df'])} 件の論文を取得しました。")