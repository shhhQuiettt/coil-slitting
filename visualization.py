import numpy as np
import tempfile
from pymoo.visualization.scatter import Scatter
from anytree.exporter import DotExporter
import matplotlib.pyplot as plt
from io import BytesIO
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
    generate_slicing_tree_img_file,
)

objective_functions = (
    average_rectangle_size,
    average_weighted_worst_percentile,
)


@st.cache_data
def load():
    sheet = load_data("./data.csv")[0]
    print(sheet)
    return sheet


def display_cuts_and_objectives(tree: Slit, sheet: npt.NDArray):
    container = st.container(border=True)
    with container:
        col1, col2, col3 = st.columns([3, 5, 2])
        with col1:
            # img_filename = generate_slicing_tree_img_file(tree)
            # st.image(img_filename, use_column_width=True)
            #
            #

            # save the image to the virtual file
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                DotExporter(tree).to_picture(f.name)

                st.image(f.name, use_column_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=make_rectangles_in_sheet(tree, sheet),
                    colorscale="viridis",
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

        with col3:
            # plot_slit_sheet(slitting_tree, st.session_state["sheet"])
            st.write("Objective functions")
            for i, func in enumerate(objective_functions):
                st.write(
                    f"{i+1}. {func.__name__} {func(tree, sensors_sheet=sheet):.4f}"
                )


@st.cache_data
def run_nsga2():
    return nsga2.run(load())


def main():
    st.set_page_config(layout="wide")
    single_sheet: npt.NDArray = load()
    res = run_nsga2()

    # unique_objectives, indices = np.unique(res.F, axis=0, return_index=True)
    # uniques_trees = res.X[indices, 0]

    # st.write("Number of unique objectives", len(unique_objectives))
    # st.write("Number of elements", len(res.X))
    # st.write(len(res.pop))
    # st.write(res.pop.get("X"))

    pareto = np.abs(res.F)
    population = np.abs(res.pop.get("F"))
    st.write(len(pareto))
    st.write(len(population))

    col1, col2 = st.columns([2, 1])
    with col1:
        # fig, ax = plt.subplots()
        # ax.scatter(
        #     population[:, 0], population[:, 1], c="lightblue", label="Population"
        # )
        # ax.scatter(pareto[:, 0], pareto[:, 1], c="red", label="Pareto front")
        # ax.set_xlabel("Average weighted worst percentile")
        # ax.set_ylabel("Average rectangle size")
        # ax.legend()
        # fig.set_figwidth(5)
        # display as plotly
        # make the same graph but in plotly

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=population[:, 0],
                y=population[:, 1],
                mode="markers",
                marker=dict(size=10, color="lightblue"),
                name="Population",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pareto[:, 0],
                y=pareto[:, 1],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Pareto front",
            )
        )
        # perfect solution
        fig.add_trace(
            go.Scatter(
                x=(np.min(pareto[:, 0]),),
                y=(np.max(pareto[:, 1]),),
                mode="markers",
                marker_symbol="star",
                # size of symbo
                marker=dict(size=20, color="green"),
                name="Hypothetical perfect solution",
            )
        )
        fig.update_layout(
            xaxis_title="Average weighted worst percentile",
            yaxis_title="Average rectangle size",
            width=900,
            height=600,
        )

        st.plotly_chart(fig)

        # st.pyplot(fig)

    # later pareto front
    # for tree, objectives in zip(uniques_trees, unique_objectives):
    for tree, objectives in zip(res.X[:, 0], res.F):
        try:
            plot_slit_sheet(tree, single_sheet)
            display_cuts_and_objectives(tree, single_sheet)
            # st.write(objectives)
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
