import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy.typing as npt

NO_OF_SENSORS = 10


def load_data(filename: str) -> npt.NDArray[np.float64]:
    data = np.genfromtxt(filename, delimiter=";", dtype=str, encoding="utf-8")
    data = np.char.replace(data, ",", ".")

    sheet = np.empty((0, data.shape[1]), dtype=np.float64)

    sheets = []
    for row in data:
        if row[0] == "":
            if sheet.shape[0] > 0:
                sheets.append(sheet)
                sheet = np.empty((0, data.shape[1]), dtype=np.float64)
            continue

        sheet = np.vstack([sheet, np.array(row, dtype=np.float64)])

    return np.array(sheets)


def display_sheet(sheet: npt.NDArray[np.float64]) -> None:
    plt.imshow(sheet)
    plt.show()


data = load_data("data.csv")

display_sheet(data[0])
