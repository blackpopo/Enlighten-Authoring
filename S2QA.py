from tqdm import tqdm
tqdm.pandas()
from streamlit_utils import *
from research_papers import research_papers
from topk_review import topk_review_papers
from cluster_review import cluster_review_papers

def display_query():
    #Queryの管理
    display_title()

    if 'query' in st.session_state:
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
    return query

def app():
    # refresh_button = st.button('Refresh button')
    # if refresh_button:
    #     st.rerun()
    #
    # debug_mode = st.checkbox("Debug Mode", value=True)
    # if debug_mode:
    #     st.session_state['debug'] = True

    display_query()

    #年の指定ボタン
    display_year_input()
    research_papers()

    # display_spaces(3)
    # topk_review_papers()
    #コミュニティグラフによるレビュー
    cluster_review_papers()



if __name__ == "__main__":
    # load_widget_state()
    st.session_state['debug'] = False
    app()
