"""
Run command:
    streamlit run app_search.py
"""
import sys
from datetime import datetime
from pathlib import Path
import json

from loguru import logger
import streamlit as st
import torch
from PIL import Image

from src.networks import Model
from src.search import Index
from src.data import image_to_encoding
from src.data import Database, fill_neighbours
from src.nms import filter_neighbours
from src.plot import draw_windows_on_one_image
from src.heuristic import calculate_heuristic


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    logger.remove(0)
    logger_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <level>{message}</level>'
    logger.add(sys.stderr, format=logger_format)
    logger.add('logs/log_{time}.log', format=logger_format)
    logger.debug('Logger set up')
except ValueError:
    pass

logger.info(f'Using device: {DEVICE}')


def main():
    st.title('Image similarity search')

    with st.form('sidebar'):
        with st.sidebar:
            st.session_state.database_folder = st.text_input('Input Database folder', value='assets/database/inception')
            st.session_state.index_folder = st.text_input('Input Index folder', value='assets/index/inception')
            st.session_state.pictures_folder = st.text_input('Output folder', value='assets/pictures')
            submitted_load = st.form_submit_button('Load')

    with st.sidebar:
        st.session_state.threshold_iou = float(st.slider('IOU threshold', value=0.1, min_value=0.0, max_value=1.0, step=0.02))

    if submitted_load:
        st.session_state.model = Model(DEVICE)
        st.session_state.database = Database.from_folder(st.session_state.database_folder)
        st.session_state.index = Index.from_folder(st.session_state.index_folder)
        st.success('Model, Database and Index loaded')

    submitted_query = False
    with st.form('query_image'):
        if not st.session_state.get('query_progress') and st.session_state.get('model'):
            query_image = st.file_uploader('Upload query image', type=['png', 'jpeg', 'jpg', 'tif'])
            submitted_query = st.form_submit_button('Search')

    if submitted_query:
        st.session_state['query_progress'] = True
        query_image = Image.open(query_image).convert('RGB')
        query_image_embedding = image_to_encoding(st.session_state.model, image=query_image)

        neighbours = st.session_state.index.search(query_image_embedding)
        heuristic = calculate_heuristic(st.session_state.model, query_image)
        neighbours = [n for n in neighbours if n.score < heuristic]
        if not neighbours:
            st.write('No matches were found')
        else:
            neighbours = fill_neighbours(neighbours, st.session_state.database)

            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_folder = Path(st.session_state.pictures_folder, timestamp)
            save_folder.mkdir(exist_ok=True)

            window_collection = []
            for item in st.session_state.database:
                image_windows = [n for n in neighbours if n.path == item.path]
                image_windows = filter_neighbours(image_windows, threshold_iou=st.session_state.threshold_iou)
                if image_windows:
                    window_collection.append({'filepath': item.path, 'windows': [n.window.as_dict() for n in image_windows]})
                    fig = draw_windows_on_one_image(item.path, image_windows, save_folder, show=False)
                    st.write(item.path)
                    st.pyplot(fig=fig, clear_figure=False, transparent=True)

            with open(Path(save_folder, 'window_collection.json'), 'w', encoding='utf-8') as f:
                json.dump(window_collection, f, ensure_ascii=False, indent=4)
            logger.info(f'Windows coordinates dumped in JSON')

        st.session_state['query_progress'] = False


if __name__ == '__main__':
    main()
