import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from data import load_data
import numpy.typing as npt
import plotly.graph_objects as go
import cv2
from slicing_tree import (
    Slit,
    average_weighted_worst_percentile,
    average_rectangle_size,
    generate_random_slitting_tree,
    plot_slit_sheet,
    make_rectangles_in_sheet,
)
from simple_cut import Sheet, Cut

objective_functions = (
    average_rectangle_size,
    average_weighted_worst_percentile,
)


@st.cache_data
def generate_nsga2():
    pass


@st.cache_data
def load():
    sheet = load_data("./data.csv")[0]
    print(sheet)
    return sheet


def display_cuts_and_objectives(sss: Sheet, sheet: npt.NDArray):
    container = st.container()
    with container:
        col1, col2 = st.columns([9, 1])
        with col1:
            print()
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=sss.get_sheet(),
                    colorscale="gray",
                    showscale=False,
                )
            )
            fig.update_layout(
                showlegend=False,
                yaxis=dict(scaleanchor="x", scaleratio=1),
                # dragmode="drawline",
            )

            # For streamlit reason, plotly chart has to have unique key
            display_cuts_and_objectives.plotly_key = (
                display_cuts_and_objectives.plotly_key + 1
                if hasattr(display_cuts_and_objectives, "plotly_key")
                else 0
            )
            plot = st.plotly_chart(
                fig, key=display_cuts_and_objectives.plotly_key
            )  # , use_container_width=True)

        with col2:
            # plot_slit_sheet(slitting_tree, st.session_state["sheet"])
            st.write("Objective functions")
            # for i, func in enumerate(objective_functions):
            st.write(
                f"{1}. {objective_functions[0].__name__} {sss.av_size:.4f}"
            )
            st.write(
                f"{2}. {objective_functions[1].__name__} {sss.av_percentile:.4f}"
            )


def main():
    single_sheet: npt.NDArray = load()
    print(single_sheet.shape)
    ##### sheet 1
    sss = Sheet(single_sheet)
    sheets_cut = []

    for i in range(2,len(single_sheet),2):
        sss.add_cut(Cut(i,-1))

    # for j in range(20,len(single_sheet[0]),20):
    #     sss.add_cut(Cut(-1, j))

    sss.av_size =  sss.average_rectangle_size()
    sss.av_percentile = sss.average_weighted_worst_percentile()
    sss.add_cuts_to_sheet()
    sss.sheet = cv2.resize(sss.sheet, (sss.sheet.shape[1]*4, sss.sheet.shape[0]*4))
    display_cuts_and_objectives(sss, sss.sheet)

    ##### sheet 2
    sss = Sheet(single_sheet)

    

    sheets_cut = []

    # for i in range(2,len(single_sheet),2):
    #     sss.add_cut(Cut(i,-1))

    for j in range(20,len(single_sheet[0]),20):
        sss.add_cut(Cut(-1, j))

    sss.av_size =  sss.average_rectangle_size()
    sss.av_percentile = sss.average_weighted_worst_percentile()
    sss.add_cuts_to_sheet()
    sss.sheet = cv2.resize(sss.sheet, (sss.sheet.shape[1]*4, sss.sheet.shape[0]*4))
    display_cuts_and_objectives(sss, sss.sheet)

    ##### sheet 3
    sss = Sheet(single_sheet)
    sheets_cut = []
    for i in range(2,len(single_sheet),4):
        sss.add_cut(Cut(i,-1))

    for j in range(20,len(single_sheet[0]),40):
        sss.add_cut(Cut(-1, j))
    sss.av_size =  sss.average_rectangle_size()
    sss.av_percentile = sss.average_weighted_worst_percentile()
    sss.add_cuts_to_sheet()
    sss.sheet = cv2.resize(sss.sheet, (sss.sheet.shape[1]*4, sss.sheet.shape[0]*4))
    display_cuts_and_objectives(sss, sss.sheet)

    ##### sheet 4
    sss = Sheet(single_sheet)
    sheets_cut = []

    sss.add_cut(Cut(5,-1))
    
    for j in range(20,len(single_sheet[0]),80):
        sss.add_cut(Cut(-1, j))
    sss.av_size =  sss.average_rectangle_size()
    sss.av_percentile = sss.average_weighted_worst_percentile()
    sss.add_cuts_to_sheet()
    sss.sheet = cv2.resize(sss.sheet, (sss.sheet.shape[1]*4, sss.sheet.shape[0]*4))
    display_cuts_and_objectives(sss, sss.sheet)

    ##### sheet 5
    sss = Sheet(single_sheet)
    sheets_cut = []

    sss.add_cut(Cut(5,sss.sheet.shape[1]//2))

    sss.av_size =  sss.average_rectangle_size()
    sss.av_percentile = sss.average_weighted_worst_percentile()
    sss.add_cuts_to_sheet()
    sss.sheet = cv2.resize(sss.sheet, (sss.sheet.shape[1]*4, sss.sheet.shape[0]*4))
    display_cuts_and_objectives(sss, sss.sheet)


if __name__ == "__main__":
    main()
