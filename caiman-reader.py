import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
import os
import tifffile
import numpy as np
import h5py
from scipy.sparse import csc_matrix
import skimage.transform 
from scipy.spatial import ConvexHull
from tkinter import ttk
from tkinter import font
import csv
import sys
import joblib
import filters

initpath = os.path.dirname(os.path.abspath(__file__))

# python caiman-reader.py --overlap-filter  -> also apply filters.ROISet
USE_OVERLAP_FILTER = "--overlap-filter" in sys.argv

# Must match OLL_Processing/oll_caiman_segmentation.py so the dF/F shown here
# is the same one saved to traces_dff.npy
F0_PERCENTILE = 8.0
F0_FLOOR = 1e-6

h5_path = None
root = tk.Tk()
root.title("CaImAn Reader")

ico_im = Image.open(os.path.join(initpath, "cmrd.ico"))
root.wm_iconphoto(True, ImageTk.PhotoImage(ico_im))
############################################################################################################
def find_movie(fold):
    """Movie for a plane folder: a local *preprocessed.tif if present, otherwise
    the input_tif recorded in the OLL pipeline's run_parameters.txt."""
    for f in sorted(os.listdir(fold)):
        if f.lower().endswith(('preprocessed.tif', 'preprocessed.tiff')):
            return os.path.join(fold, f)
    run_params = os.path.join(fold, "run_parameters.txt")
    if os.path.exists(run_params):
        with open(run_params, "r") as f:
            for line in f:
                key, _, value = line.partition("=")
                if key.strip() == "input_tif":
                    return value.strip()
    return None


def quality_csv_path(h5_file):
    """results.hdf5 -> results_quality.csv in the same folder."""
    return os.path.splitext(h5_file)[0] + "_quality.csv"


def load_quality_metric(h5, fold, h5_key, npz_key, n_components):
    """Per-component SNR / r-value. Falls back to the OLL pipeline's
    quality_scores.npz when evaluate_components failed and CaImAn saved None."""
    ds = h5["estimates"].get(h5_key)
    if isinstance(ds, h5py.Dataset) and ds.ndim == 1:
        return list(ds[()])
    npz = os.path.join(fold, "quality_scores.npz")
    if os.path.exists(npz):
        return list(np.load(npz)[npz_key])
    return [float("nan")] * n_components


def initialize_project():
    global welcome, active_vid, folder, footprints, radiocanvas, scrollable_buttons, quality_check, traces, snr, rval, root, h5_path

    fold = filedialog.askdirectory(initialdir=initpath, title="Select Folder")
    if not fold:
        return

    # Only look in the selected plane folder, not its subfolders
    h5_files = sorted(os.path.join(fold, f) for f in os.listdir(fold) if f.endswith('.hdf5'))
    if not h5_files:
        messagebox.showerror("CaImAn Reader", f"No .hdf5 file found in:\n{fold}")
        return
    h5_file = h5_files[0]

    tif = find_movie(fold)
    if tif is None or not os.path.exists(tif):
        messagebox.showerror("CaImAn Reader", f"Movie not found for:\n{fold}\n\nLooked for *preprocessed.tif and run_parameters.txt input_tif:\n{tif}")
        return
    print(tif)

    size_up = 2

    # Load into locals first so a failed load leaves the current project intact
    with h5py.File(h5_file, "r") as h5:
        n_components = h5['estimates']['C'].shape[0]
        new_snr = load_quality_metric(h5, fold, "SNR_comp", "snr", n_components)
        new_rval = load_quality_metric(h5, fold, "r_values", "rval", n_components)

        # Default: review every component CaImAn accepted (estimates/idx_components).
        # --overlap-filter additionally applies filters.ROISet (r-value prepass, then
        # drops the weaker of overlapping, correlated ROI pairs)
        rois_to_use = None
        if USE_OVERLAP_FILTER:
            roiset = filters.ROISet(h5, F_dff=load_dff(h5), snr=new_snr, rval=new_rval)
            roiset.prepass(rval_thresh=0.3)
            roiset.binarize()
            roiset.find_overlap()
            rois_to_use = roiset.filter_high_overlap(corr_thresh=0.1, overlap=0.45)
            print(f"Overlap filter kept {len(rois_to_use)} of {len(roiset.good_idxs)} accepted components")

        new_footprints, new_traces = generate_footprints(h5, scale=size_up, rois_to_use=rois_to_use)

    if not new_footprints:
        messagebox.showerror("CaImAn Reader", f"No components to review in:\n{fold}")
        return

    for rb in scrollable_buttons.winfo_children():
        rb.destroy()
    folder.set(fold)
    h5_path = h5_file
    footprints, traces, snr, rval = new_footprints, new_traces, new_snr, new_rval
    active_vid = load_tiffstack(tif, size_up)

    quality_check = {int(k): '' for k in footprints.keys()}
    quality_csv = quality_csv_path(h5_file)
    if not os.path.exists(quality_csv):
        print("NEW QUALITY CSV")
    else:
        with open(quality_csv, "r") as f:
            r = csv.reader(f)
            for pair in r:
                k, v = pair
                if int(k) in quality_check:
                    quality_check[int(k)] = v


    active_cell.set(int(list(footprints.keys())[0]))
    update_mpl()
    tif2frame(active_vid, 0)
    set_scale(active_vid)
    new_cells(footprints, quality_check)
    update_readout()
    if not root.winfo_ismapped():
        root.deiconify()
    welcome.destroy()


def load_tiffstack(path2tiff, size_up):

    def resize_bitcrunch(pc, xdim, ydim, frame_dtype):
        resized = [skimage.transform.resize(i, 
                                    (xdim, ydim),
                                     anti_aliasing=True,
                                     preserve_range=True).astype(frame_dtype) for i in pc]
        
        bitcrunch = [np.interp(resized[i], (resized[i].min(), resized[i].max()), (0,255)).astype(np.uint8) 
        for i in range(len(resized))]
        return bitcrunch
        
    if path2tiff != None:
        with tifffile.TiffFile(path2tiff) as tiff:
            pc = tiff.asarray(out="memmap")
            
        #test resize:
        xdim = pc[0].shape[0]*size_up
        ydim = pc[0].shape[1]*size_up
        frame_dtype =  pc[0].dtype


        numchunks = int(pc.shape[0]/20)
        slices = list(range(0, pc.shape[0], numchunks))

        refslice = []

        for n in range(len(slices) - 1):
            refslice.append([slices[n], slices[n+1]])
        refslice.append([slices[-1], pc.shape[0]])

        vidslices = joblib.Parallel(n_jobs=20)(joblib.delayed(resize_bitcrunch)(pc[i[0]:i[1]], xdim, ydim, frame_dtype) for i in refslice)

    bitcrunch = np.concatenate(vidslices)
    return bitcrunch
    

def tif2frame(arrtiff, i):
    global tifviz
    aframe = arrtiff[i]
    height, width = aframe.shape
    tifviz.config(height=height, width=width)
    img = ImageTk.PhotoImage(Image.fromarray(aframe))
    tifviz.create_image(0,0, anchor=tk.NW, image=img)
    update_overlay()
    tifviz.image = img


def set_scale(listtiff):
    global frame_scroll
    new_max = len(listtiff)-1
    new_width = listtiff[0].shape[1] - 30
    frame_scroll.config(to=int(new_max), resolution=1, length=new_width)


def scale_img(index):
    global active_vid
    tif2frame(active_vid, int(index))
    mpl_scan(int(index))


def play_pause():
    global playing
    playing = not playing
    if playing:
        pp_button.config(image=pause)
        update_slider()
    else:
        pp_button.config(image=play)


def update_slider():
    global playing, frame_scroll, active_vid
    if playing:
        current = int(frame_scroll.get())  
        next_frame = current + 1

        if next_frame >= len(active_vid):  
            playing = False
        else:
            frame_scroll.set(next_frame)  
            root.after(20, update_slider)  


def load_dff(h5):
    """estimates/F_dff if saved. CaImAn saves it as the string 'NoneType' unless
    detrend_df_f() was run; in that case compute dF/F from C the same way the
    OLL pipeline does."""
    C = np.array(h5['estimates']['C'])
    F_dff = h5['estimates'].get('F_dff')
    if isinstance(F_dff, h5py.Dataset) and F_dff.shape == C.shape:
        return np.array(F_dff)
    F0 = np.nanpercentile(C, F0_PERCENTILE, axis=1, keepdims=True)
    F0 = np.where(np.isfinite(F0) & (F0 > F0_FLOOR), F0, F0_FLOOR)
    return (C - F0) / F0


def generate_footprints(h5, scale=1, rois_to_use=None):
    if rois_to_use is not None:
        cells = [int(i) for i in rois_to_use]
    else:    
        cells = [int(i) for i in list(h5['estimates']["idx_components"])]
    data = h5['estimates']['A']['data']
    indices = h5['estimates']['A']['indices']
    indptr = h5['estimates']['A']['indptr']
    shape = h5['estimates']['A']['shape']
    y_dims = h5['estimates']['dims'][1]
    x_dims = h5['estimates']['dims'][0]

    C = np.array(h5['estimates']['C'])
    YrA = np.array(h5['estimates']['YrA'])
    F_dff = load_dff(h5)
    raw_traces = np.zeros(C.shape)
    
    for i, j in enumerate(C):    
        raw_traces[i,:] = np.add(YrA[i,:], j)
    
    traces = {k:[raw_traces[k], F_dff[k]] for k in range(0,len(C))}
    if not cells:
        return {}, traces

    

    
    sparse_mtx = csc_matrix((data, indices, indptr), shape=shape)
    feet = sparse_mtx.toarray()

    feet = np.array([feet[:,i] for i in cells]).T
    img_bounds = {int(cell): [] for cell in cells}

    for j in range(feet.shape[1]):

        onefeet = np.reshape(feet[:, j],  (y_dims,  x_dims)).T 

        bounds = []
        onefeet= skimage.transform.rescale(onefeet, scale, anti_aliasing=True) 
        pxlft = np.argwhere(onefeet!=0)
        bounds = np.asarray(pxlft)
        hull = ConvexHull(bounds)
        img_bounds[cells[j]] = bounds[hull.vertices]
        
    
    flat_bounds = {}
    for k, v in img_bounds.items():
        if len(v) > 0:
            flat_bounds[k] = [j for i in v for j in reversed(i.tolist())]
        else:
            pass
    
    return flat_bounds, traces



def update_overlay():
    global active_cell, tifviz, footprints, quality_check, quality, hidecell, alla, allr, allf, allu
    cell = active_cell.get()
    ft_list = footprints[cell]
    tifviz.delete("overlay")
    if not hidecell.get():
        if not (alla.get() or allr.get() or allf.get() or allu.get()):
            tifviz.create_polygon(ft_list, fill="", outline="red", tags="overlay")
        else:
            toggle_all()

    quality.set(quality_check[int(cell)]) 


def new_cells(footprints, quality_check):
    global scrollable_buttons
    quality_color = {"A":"#198114", "R":"#443F9E", "F":"#ac0f0f", "":"#000000"}
    for cell in footprints:
        if quality_check[cell] == '':
            tk.Radiobutton(scrollable_buttons, text=str(cell), variable=active_cell, font=font.Font(underline=True, size=16), foreground=quality_color[quality_check[cell]], value=cell, command=lambda: [update_mpl(), update_overlay()]).pack()
        else:   
            tk.Radiobutton(scrollable_buttons, text=str(cell), variable=active_cell, font=font.Font(size=16), value=cell, foreground=quality_color[quality_check[cell]], command=lambda: [update_mpl(), update_overlay()]).pack()

def mousewheel(e):
    radiocanvas.yview_scroll(-1 * (e.delta // 120), "units")

def update_qualdict():
    global quality, quality_check, save_status
    review_confirmed(int(active_cell.get()))
    quality_check[int(active_cell.get())] = str(quality.get())
    update_readout()
    save_status.config(image=klaxon)

def review_confirmed(uncheck):
    global scrollable_buttons, quality
    quality_color = {"A":"#198114", "R":"#443F9E", "F":"#ac0f0f", "":"#000000"}
    for rb in scrollable_buttons.winfo_children():
        if isinstance(rb, tk.Radiobutton) and (int(rb.cget("value")) == uncheck):
            rb.config(font=font.Font(size=16), foreground=quality_color[quality.get()])




def save_project():
    global quality_check, save_status
    if h5_path is None:
        return
    name = quality_csv_path(h5_path)
    with open(name, "w", newline='') as f:
        w = csv.writer(f)
        for k, v in quality_check.items():
            w.writerow([k, v])
    save_status.config(image=check)

def next_cell():
    global active_cell, footprints, quality, quality_check
    fpts = [i for i in footprints.keys()]
    nt = fpts.index(int(active_cell.get())) + 1
    active_cell.set(fpts[nt])
    quality.set(quality_check[int(active_cell.get())]) 
    update_overlay()
    update_mpl()
    update_readout()

def previous_cell():
    global active_cell, footprints, quality, quality_check
    fpts = [i for i in footprints.keys()]
    nt = fpts.index(int(active_cell.get())) - 1
    active_cell.set(fpts[nt])
    quality.set(quality_check[int(active_cell.get())])
    update_overlay() 
    update_mpl()
    update_readout()

def update_mpl():
    global traces, ax, mpl_canvas, active_cell, frame_scroll, vertical_line
    raw_trace = traces[int(active_cell.get())][0]
    fdf = traces[int(active_cell.get())][1]
    ax.cla()
    ax.plot(raw_trace, alpha=0.5)
    ax.plot(fdf, alpha=0.8)
    vertical_line = None
    mpl_scan(float(frame_scroll.get()))
    mpl_canvas.draw()
    update_readout()


def mpl_scan(i):
    global ax, vertical_line
    if vertical_line:
        vertical_line.set_xdata([float(i)])
    else:
        vertical_line = ax.axvline(x=float(i), color='r')
    mpl_canvas.draw()


def update_readout():
    global active_cell, quality, readout, quality_check
    qkey = {"A":"Accepted", "R": "Rejected", "F":"Flagged", '':"Unreviewed"}
    overall_status = [*quality_check.values()]

    rout = f"""Cell ID: {active_cell.get()}
Quality label: {qkey[quality.get()]}
SNR: {round(snr[active_cell.get()], 3)}
Spatial Correlation: {round(rval[active_cell.get()], 3)}
Number of Cells:
Total: {len(overall_status)} Accepted: {overall_status.count("A")} Rejected: {overall_status.count("R")} 
Flagged: {overall_status.count("F")} Unreviewed: {overall_status.count("")}
"""
    readout.config(text=rout)

def toggle_all():
    global quality_check, alla, allr, allf, footprints, quality, active_cell
    tifviz.delete("overlay")
    disp_a = disp_r = disp_f = disp_u = []
    if alla.get():
        disp_a = [c for c, r in quality_check.items() if r=="A"]
    if allr.get():
        disp_r = [c for c, r in quality_check.items() if r=="R"]
    if allf.get():
        disp_f = [c for c, r in quality_check.items() if r=="F"]
    if allu.get():
        disp_u = [c for c, r in quality_check.items() if r==""]    
    disps = disp_a + disp_r + disp_f + disp_u 
    for i in disps:
        ft = footprints[i]
        tifviz.create_polygon(ft, fill="", outline="red", tags="overlay")
    if not (alla.get() or allr.get() or allf.get() or allu.get()):
        tifviz.create_polygon(footprints[active_cell.get()], fill="", outline="red", tags="overlay")

def single_switch():
    global active_cell, footprints
    if hidecell.get():
        tifviz.delete("overlay")
    else:
        if (alla.get() or allr.get() or allf.get() or allu.get()):
            toggle_all()
        else:    
            tifviz.create_polygon(footprints[active_cell.get()], fill="", outline="red", tags="overlay") 

def allinone():
    global allaccepted, allrejected, allflagged, allunrev, allall
    if allall.get():
        allaccepted.select()
        allflagged.select()
        allunrev.select()
        allrejected.select()
        toggle_all()
    else:
        allaccepted.deselect()
        allflagged.deselect()
        allunrev.deselect()
        allrejected.deselect()
        toggle_all()
        


# icons
klaxon = ImageTk.PhotoImage(Image.open(os.path.join(initpath, "icons/klaxon.png")).resize((15,15)))
right_arr = ImageTk.PhotoImage(Image.open(os.path.join(initpath, "icons/right_arrow.png")).resize((15,15)))
left_arr = ImageTk.PhotoImage(Image.open(os.path.join(initpath, "icons/left_arrow.png")).resize((15,15)))
check = ImageTk.PhotoImage(Image.open(os.path.join(initpath, "icons/check.png")).resize((15,15)))
pause = ImageTk.PhotoImage(Image.open(os.path.join(initpath, "icons/pause.png")).resize((15,15)))
play = ImageTk.PhotoImage(Image.open(os.path.join(initpath, "icons/play.png")).resize((15,15)))
from_load = ImageTk.PhotoImage(Image.open(os.path.join(initpath, "icons/from_load.png")).resize((15,15)))

####################################################################################

# Main Window

snr = []
rval =[]
vertical_line = None
traces = {}
footprints={}
quality_check = {}
playing=False
viz = tk.Frame(root)
radioframe = tk.Frame(root)
eval_frame = tk.Frame(root)
projb_frame = tk.Frame(root)
mpl_frame = tk.Frame(root)
radiocanvas = tk.Canvas(radioframe, height=220, width=80)
readoutfr = tk.Frame(root)

#Eval buttons:


quality = tk.StringVar()
accept = tk.Radiobutton(eval_frame, text="Accept", variable=quality, value="A", foreground="#198114", command=update_qualdict)
reject = tk.Radiobutton(eval_frame, text="Reject", variable=quality, value="R", foreground="#443F9E", command=update_qualdict)
flg_ = tk.Radiobutton(eval_frame, text="Flag", variable=quality, value="F", foreground="#ac0f0f", command=update_qualdict)

prvs = tk.Button(eval_frame, image=left_arr, command=previous_cell)
nxt = tk.Button(eval_frame, image=right_arr, command=next_cell)

accept.grid(row=1, column=0)
reject.grid(row=1, column=1)
flg_.grid(row=1, column=2)
prvs.grid(row=2, column=0)
nxt.grid(row=2, column=2)

#Overall Project Buttons:
save_button = tk.Button(projb_frame, text="Save Quality Values", command=save_project)
save_status = tk.Label(projb_frame, image=from_load)
new_project = tk.Button(projb_frame, text="Analyze New Video", command=initialize_project)
save_button.grid(row=0, column=0)
save_status.grid(row=0, column=1)
new_project.grid(row=0, column=2)


##########################################################
# Matplotlib widgets

fig = Figure(figsize=(7,1.5))  #ADD DIMENSIONS/DPI
ax = fig.add_subplot()
mpl_canvas = FigureCanvasTkAgg(fig, master=mpl_frame)
mpl_canvas.get_tk_widget().pack()

tools = NavigationToolbar2Tk(mpl_canvas, mpl_frame)
tools.update()
mpl_canvas.get_tk_widget().pack()
##########################################################


readout = tk.Label(readoutfr, anchor="e", justify="left")
tifviz = tk.Canvas(viz, width=1, height=1)
tifviz.grid(row=1, columnspan=2)
frame_scroll = tk.Scale(viz, from_=0, to=1, orient="horizontal", command=scale_img)
frame_scroll.grid(row=0, column=1)

pp_button = tk.Button(viz, image=play, command=play_pause)
scrollbar = ttk.Scrollbar(radioframe, orient="vertical", command=radiocanvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
scrollable_buttons = tk.Frame(radiocanvas, height=600, width=80)
subwindow = radiocanvas.create_window((0,0), window=scrollable_buttons, anchor="nw")
radiocanvas.configure(yscrollcommand=scrollbar.set)
scrollable_buttons.bind("<Configure>", lambda e: radiocanvas.configure(scrollregion=radiocanvas.bbox("all")))

radiocanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
radiocanvas.bind_all("<MouseWheel>", mousewheel)

pp_button.grid(row=0, column=0)

###############################################################################
# Draw from cr HOLD
#Eval buttons:
hidecell = tk.BooleanVar(value=False)
quality = tk.StringVar()

toggle_cellview = tk.Checkbutton(eval_frame, text="Toggle ROI view off", onvalue=True, offvalue=False, variable=hidecell, command=single_switch)
toggle_cellview.grid(row=0, column=0, columnspan=2)

accept = tk.Radiobutton(eval_frame, text="Accept", variable=quality, value="A", foreground="#198114", command=update_qualdict)
reject = tk.Radiobutton(eval_frame, text="Reject", variable=quality, value="R", foreground="#443F9E", command=update_qualdict)
flg_ = tk.Radiobutton(eval_frame, text="Flag", variable=quality, value="F", foreground="#ac0f0f", command=update_qualdict)

prvs = tk.Button(eval_frame, image=left_arr, command=previous_cell)
nxt = tk.Button(eval_frame, image=right_arr, command=next_cell)


accept.grid(row=1, column=0)
reject.grid(row=1, column=1)
flg_.grid(row=1, column=2)
prvs.grid(row=2, column=0)
nxt.grid(row=2, column=2)

#Overall Project Buttons:
save_button = tk.Button(projb_frame, text="Save Quality Values", command=save_project)
save_status = tk.Label(projb_frame, image=from_load)
new_project = tk.Button(projb_frame, text="Analyze New Video", command=initialize_project)
save_button.grid(row=0, column=0)
save_status.grid(row=0, column=1)
new_project.grid(row=0, column=2)


# Toggle view all:
allall = tk.BooleanVar()
alla = tk.BooleanVar()
allr = tk.BooleanVar()
allf = tk.BooleanVar()
allu = tk.BooleanVar()

readout.grid(row=0)
allaccepted = tk.Checkbutton(readoutfr, text="Accepted ROIs", variable=alla, onvalue=True, offvalue=False, command=toggle_all)
allrejected = tk.Checkbutton(readoutfr, text="Rejected ROIs", variable=allr, onvalue=True, offvalue=False, command=toggle_all)
allflagged = tk.Checkbutton(readoutfr, text="Flagged ROIs", variable=allf, onvalue=True, offvalue=False, command=toggle_all)
allunrev = tk.Checkbutton(readoutfr, text="Unreviewed ROIs", variable=allu, onvalue=True, offvalue=False, command=toggle_all)
showall = tk.Checkbutton(readoutfr, text="Display all:", variable=allall, onvalue=True, offvalue=False, command=allinone)
showall.grid(row=1)
allaccepted.grid(row=2)
allrejected.grid(row=3)
allflagged.grid(row=4)
allunrev.grid(row=5)



###############################################################################

#Frame organization
readoutfr.grid(row=0, column=0)
mpl_frame.grid(row=0, column=1)
radioframe.grid(row=1, column=0)
viz.grid(row=1, column=1)
eval_frame.grid(row=2, column=0)
projb_frame.grid(row=2, column=1)


active_cell = tk.IntVar()
folder = tk.StringVar(value="No Folder selected")
root.withdraw()


####################################################################################

welcome = tk.Toplevel(root)
welcome.title("Welcome to CaImAn Reader")

logo = Image.open(os.path.join(initpath, "logo.png")).resize((50,50))
logo_rs = ImageTk.PhotoImage(logo)
welcome_note = tk.Label(welcome, text="Welcome to CaImAn Reader!  \nChoose a plane folder (e.g. Segmentation_caImAn/<date>/<mouse>/plane0) to begin")
file_b = tk.Button(welcome, text="Browse Files", command=initialize_project)

show_logo = tk.Label(welcome, image=logo_rs)
show_logo.pack()
welcome_note.pack()
file_b.pack()

####################################################################################

root.mainloop()

