from numpy import size
import numpy as np
import streamlit as st
from data import load_data
import numpy.typing as npt
import plotly.graph_objects as go
import cv2
import nsga2
from slicing_tree import (
    Slit,
    average_weighted_worst_percentile,
    average_rectangle_size,
    generate_random_slitting_tree,
    plot_slit_sheet,
    make_rectangles_in_sheet,
)

objective_functions = (
    average_rectangle_size,
    average_weighted_worst_percentile,
)


@st.cache_data
def generate_nsga2():
    return nsga2.run()


@st.cache_data
def load():
    sheet = load_data("./data.csv")[0]
    print(sheet)
    return sheet


def display_cuts_and_objectives(tree: Slit, sheet: npt.NDArray):
    container = st.container()
    with container:
        col1, col2 = st.columns([9, 1])
        with col1:
            print()
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=make_rectangles_in_sheet(tree, sheet),
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
            for i, func in enumerate(objective_functions):
                st.write(
                    f"{i+1}. {func.__name__} {func(tree, sensors_sheet=sheet):.3f}"
                )


def main():
    single_sheet: npt.NDArray = load()
    res = nsga2.run(single_sheet)

    print(type(res.F))

    unique_objectives, indices = np.unique(res.F, axis=0, return_index=True)
    uniques_trees = res.X[indices, 0]

    st.write("Number of unique objectives", len(unique_objectives))

    # later pareto front
    for tree, objectives in zip(uniques_trees, unique_objectives):
        try:
            display_cuts_and_objectives(tree, single_sheet)
        except IndexError:
            st.write("Could not process tree")

    # st.set_page_config(layout="wide")

    # sheet_size = single_sheet.shape[::-1]
    # size_multiplier = 1
    # new_size = (sheet_size[0] * size_multiplier, sheet_size[1] * size_multiplier)
    # slitting_tree = generate_random_slitting_tree(2)
    # single_sheet = cv2.resize(single_sheet, new_size)

    # display_cuts_and_objectives(slitting_tree, single_sheet)

    # slitting_tree2 = generate_random_slitting_tree(3)
    # display_cuts_and_objectives(slitting_tree2, single_sheet)


if __name__ == "__main__":
    main()
