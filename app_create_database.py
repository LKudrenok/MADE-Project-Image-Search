"""
Run command:
    streamlit run app_create_database.py
"""
import sys

from loguru import logger
import streamlit as st
import torch

from src.networks import Model
from src.search import Index
from src.data import create_database
from src.detection import Parameters


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
    st.title('Database & Index preparation')

    with st.form('sidebar'):
        with st.sidebar:
            st.session_state.images_folder = st.text_input('Input images folder', value='assets/images/base_images')
            st.session_state.database_folder = st.text_input('Output Database folder', value='assets/database')
            st.session_state.index_folder = st.text_input('Output Index folder', value='assets/index')

            st.session_state.batch_size = int(st.number_input('Batch size', min_value=1, value=64))
            st.session_state.scale = float(st.number_input('Scale', min_value=0.1, value=1.5, step=0.1))
            st.session_state.step_size = int(st.number_input('Step size', min_value=1, value=20))
            st.session_state.win_width = int(st.number_input('Window width', min_value=1, value=40))
            st.session_state.win_height = int(st.number_input('Window height', min_value=1, value=40))
            st.session_state.min_size_factor = int(st.number_input('Min size factor', min_value=1, value=2))

            submitted_create = st.form_submit_button('Create')

    if submitted_create:
        detection_params = Parameters(scale=st.session_state.scale,
                                      step_size=st.session_state.step_size,
                                      win_width=st.session_state.win_width,
                                      win_height=st.session_state.win_height,
                                      min_size_factor=st.session_state.min_size_factor)
        st.session_state.model = Model(DEVICE)
        st.success('Model loaded')

        st.info('Creating Database...')
        st.session_state.database = create_database(st.session_state.images_folder,
                                                    model=st.session_state.model,
                                                    progress_bar=True,
                                                    batch_size=st.session_state.batch_size,
                                                    parameters=detection_params)
        st.session_state.database.save(folder=st.session_state.database_folder)
        st.success(f'Database built and saved ({len(st.session_state.database)} images)')

        st.info('Creating Index...')
        st.session_state.index = Index()
        st.session_state.index.build(st.session_state.database.embeddings)
        st.session_state.index.save(folder=st.session_state.index_folder)
        st.success('Index built and saved')


if __name__ == '__main__':
    main()
