from data import load_data, display_sheet
import numpy as np

single_sheet = load_data("./data.csv")[2]

class Cut:
    def __init__(self,x, y):
        self.x = x
        self.y = y

class Sheet:

    def __init__(self, sheet: np.ndarray):
        self.sheet = self.remove_zeros(sheet)
        self.cuts = []

    def add_cut(self, cut: Cut):
        if cut.x < len(self.sheet) and cut.y < len(self.sheet[0]):
            self.cuts.append(cut)

    def remove_zeros(self, sheet: np.ndarray) -> np.ndarray:
        non_zero_columns = ~np.all(sheet == 0, axis=0)

        filtered_arr = sheet[:, non_zero_columns]

        return filtered_arr
    
    def add_cuts_to_sheet(self):
        for cut in self.cuts:
            if cut.x != -1:
                # add row to sheet on given position
                self.sheet[cut.x, :] = 0
            if cut.y != -1:
                # add column to sheet on given position
                self.sheet[:, cut.y] = 0


    def average_weighted_worst_percentile(self) -> np.float64:
        percentile = 5
        cut_x = sorted(self.cuts, key=lambda x: x.x)
        cut_y = sorted(self.cuts, key=lambda x: x.y)
        fragments = []
        results = []
        last_cut = 0
        for i in range(len(cut_x)):
            if cut_x[i].x == -1:
                continue
            
            fragments.append(self.sheet[last_cut:cut_x[i].x, :])
            last_cut = cut_x[i].x
        
        fragments.append(self.sheet[last_cut:, :])
        
        for fragment in fragments:
            last_cut = 0
            for i in  range(len(cut_y)):
                if cut_y[i].y == -1:
                    continue

                results.append(fragment[:, last_cut:cut_y[i].y])
                last_cut = cut_y[i].y
            
            results.append(fragment[:, last_cut:])
        # print(results)
        percentiles = np.array([np.percentile(result, percentile, method='averaged_inverted_cdf') for result in results])
        areas = np.array([result.shape[0]*result.shape[1] for result in results])
        
      
        return np.sum(percentiles*areas)/np.sum(areas)
    



sss = Sheet(single_sheet)


sheets_cut = []
# for i in range(1,len(single_sheet),2):
#     sss.add_cut(Cut(i,-1))

for j in range(20,len(single_sheet[0]),20):
    sss.add_cut(Cut(-1, j))

sss.average_weighted_worst_percentile()
sss.add_cuts_to_sheet()
display_sheet(sss.sheet)
