# CV Must-Know: Quick Q&A (short version)

> The **most-asked general CV questions** for a Computer Vision Engineer role, picked from what your resume signals: **OpenCV classical (heaviest)** → detection/YOLO → segmentation/SAM → embeddings → OCR → DL basics → deployment.
> Each answer has a simple explanation, code where it helps, and 🎯 **a line to say in the interview**.
> ⭐⭐⭐ = almost certain to come up · ⭐⭐ = likely · ⭐ = good to know

---

## 1. Image basics

**1.1 What is an image to a computer?** ⭐⭐⭐
A grid of numbers. Grayscale = `H × W`; colour = `H × W × 3`. Each value is usually `uint8`, from 0 (dark) to 255 (bright).
```python
img = cv2.imread("a.jpg")          # BGR, uint8, shape (H, W, 3); returns None if the path is wrong
h, w = img.shape[:2]
crop = img[y1:y2, x1:x2]           # rows (y) first, then columns (x)
```
🎯 *"An image is a NumPy array of shape H×W×C, usually uint8 from 0 to 255. OpenCV stores colour as BGR."*

**1.2 BGR vs RGB: why does it matter?** ⭐⭐⭐
OpenCV loads images as **BGR**. Matplotlib and PyTorch/ImageNet models expect **RGB**. If you forget to convert, red and blue are swapped and the model silently gets worse.
```python
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
🎯 *"OpenCV uses BGR. I convert to RGB before display or before feeding a model trained on RGB."*

**1.3 Why use different colour spaces?** ⭐⭐
| Space | Use |
|---|---|
| **Gray** | Most classical processing (edges, threshold, contours). One channel, so it's faster |
| **HSV** | Colour-based masking. Hue (the colour itself) is separate from brightness, so lighting matters less |
| **LAB** | Perceptual colour differences; CLAHE on the L channel |
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (100, 80, 50), (130, 255, 255))   # blue range
```
⚠️ OpenCV hue runs **0–179**. **Red wraps around 0/180**, so it needs two ranges combined.
🎯 *"HSV separates colour from brightness, so colour thresholds survive lighting changes."*

**1.4 Resizing: which interpolation?** ⭐⭐
- Shrinking → `INTER_AREA` (avoids aliasing)
- Enlarging → `INTER_LINEAR` (fast) or `INTER_CUBIC` (sharper)
- Masks or labels → `INTER_NEAREST` (never invents new label values)
```python
small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)   # size is (width, height)!
```
**Keep the aspect ratio:** scale by `s = target / max(h, w)`. YOLO uses **letterbox**: resize keeping the aspect ratio, then pad to a square with gray.
🎯 *"INTER_AREA to shrink, linear or cubic to enlarge, nearest for masks."*

---

## 2. OpenCV core (expect the most depth here)

### Function cheat sheet ⭐⭐⭐
| Task | Function | Remember |
|---|---|---|
| Colour convert | `cvtColor` | BGR2GRAY, BGR2HSV, BGR2RGB |
| Threshold | `threshold`, `adaptiveThreshold` | `+THRESH_OTSU` picks the threshold automatically |
| Blur | `GaussianBlur`, `medianBlur`, `bilateralFilter` | kernel size must be odd |
| Morphology | `erode`, `dilate`, `morphologyEx` | OPEN removes specks, CLOSE fills holes |
| Edges | `Sobel`, `Laplacian`, `Canny` | Sobel → `CV_64F` |
| Shapes | `findContours`, `boundingRect`, `contourArea`, `approxPolyDP`, `minAreaRect`, `moments` | input = binary, white object on black |
| Lines / circles | `HoughLinesP`, `HoughCircles` | blur first |
| Find a pattern | `matchTemplate` + `minMaxLoc` | not scale- or rotation-invariant |
| Differences | `absdiff` (+ SSIM from skimage) | align the images first |
| Features | `ORB_create`, `SIFT_create`, `BFMatcher`, `findHomography` | ORB → Hamming distance |
| Geometry | `getRotationMatrix2D`, `warpAffine`, `getPerspectiveTransform`, `warpPerspective` | affine needs 3 points, perspective needs 4 |
| Contrast | `equalizeHist`, `createCLAHE` | CLAHE = local and safer |
| Blobs | `connectedComponentsWithStats` | label 0 = background |
| Masks | `inRange`, `bitwise_and/or/not` | |
| Drawing | `rectangle`, `circle`, `putText`, `drawContours` | colours are BGR |
| Video | `VideoCapture`, `VideoWriter` | `ret, frame = cap.read()` |

---

**2.1 What is convolution / a kernel?** ⭐⭐⭐
A small matrix (for example 3×3) slides over the image. At each position it multiplies the values underneath it, adds them up, and writes one output pixel. The kernel's numbers decide the effect: averaging → blur, differences → edges.
```python
sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
out = cv2.filter2D(img, -1, sharpen)
```
🎯 *"Convolution slides a small weighted window over the image; the weights decide whether it blurs, sharpens or finds edges."*

**2.2 Thresholding: simple vs Otsu vs adaptive?** ⭐⭐⭐
- **Simple:** one fixed value for the whole image.
- **Otsu:** picks the threshold automatically, for histograms with two peaks (object vs background).
- **Adaptive:** a different threshold per neighbourhood, for **uneven lighting** such as shadows or scans.
```python
_, bw   = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
```
Use `THRESH_BINARY_INV` for dark objects on a light background (for example black lines on a white drawing), so the objects become white for `findContours`.
🎯 *"Otsu when lighting is even and the histogram is bimodal, adaptive when lighting varies."*

**2.3 Gaussian vs median vs bilateral blur?** ⭐⭐⭐
| Filter | What it does | Best for |
|---|---|---|
| **Gaussian** | Weighted average; nearby pixels count more | General noise; the usual step before edges |
| **Median** | Takes the median of the window | **Salt-and-pepper** noise; keeps edges fairly well |
| **Bilateral** | Averages only pixels that are close **and** similar in colour | Smoothing while **keeping edges sharp** (slower) |
```python
g = cv2.GaussianBlur(img, (5, 5), 0)
m = cv2.medianBlur(img, 5)
b = cv2.bilateralFilter(img, 9, 75, 75)
```
🎯 *"Gaussian for general noise, median for salt-and-pepper, bilateral when edges must stay sharp."*

**2.4 Erosion, dilation, opening, closing?** ⭐⭐⭐
These work on binary masks (white = object).
- **Erode:** shrinks white areas and removes thin specks.
- **Dilate:** grows white areas and joins gaps.
- **Opening** = erode then dilate → **removes small white noise** while keeping the object's size.
- **Closing** = dilate then erode → **fills small holes or gaps** inside objects.
```python
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
clean  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
```
🎯 *"Opening removes specks, closing fills holes. Both clean up a mask before finding contours."*

**2.5 Sobel: what is it and how do you use it?** ⭐⭐⭐
An edge is a **sudden change in brightness**. Sobel measures that change (the gradient) in x and in y. Its kernel combines a derivative with a little smoothing:
`Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]`
```python
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # change along x → finds vertical edges
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # change along y → finds horizontal edges
mag = cv2.magnitude(gx, gy)                       # edge strength = sqrt(gx² + gy²)
show = cv2.convertScaleAbs(mag)                   # back to uint8 for display
```
⚠️ **Why `CV_64F`?** Gradients can be **negative** (dark→bright vs bright→dark). With `uint8` the negatives are clipped to 0, so you lose half the edges.
- Edge direction = `atan2(gy, gx)`.
🎯 *"Sobel computes the brightness gradient in x and y; the magnitude gives edge strength. I use a float output so negative gradients aren't clipped."*

**2.6 Canny: steps and thresholds?** ⭐⭐⭐
Canny gives **thin, clean, 1-pixel binary edges**. It has 5 steps:
1. **Gaussian blur** to remove noise. In OpenCV you usually do this yourself before calling `Canny`.
2. **Sobel gradients**: magnitude and direction.
3. **Non-maximum suppression**: keep only the peak pixel across each edge, which thins the edges.
4. **Double threshold**: above high = strong edge; between low and high = weak edge.
5. **Hysteresis**: keep weak edges **only if they connect to strong ones**, which removes isolated noise.
```python
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)   # low:high ≈ 1:2 to 1:3
```
🎯 *"Canny is blur, gradients, non-max suppression to thin the edges, then a double threshold with hysteresis to keep real edges and drop noise."*

**2.7 Sobel vs Canny vs Laplacian?** ⭐⭐
| | Output | Note |
|---|---|---|
| **Sobel** | Gray map of gradient strength (thick edges) | 1st derivative; has direction |
| **Canny** | Thin binary edges | Most robust to noise; the usual choice |
| **Laplacian** | 2nd derivative, all directions at once | Very sensitive to noise. **Blur check:** low variance means blurry |
```python
sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()   # low → blurry (threshold depends on the data)
```
🎯 *"Sobel gives gradient strength, Canny gives clean thin edges, and the variance of the Laplacian is a quick blur detector."*

**2.8 Contours: how do you find and filter objects?** ⭐⭐⭐
A contour is the **outline of a white region** in a binary image.
```python
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) < 100:          # drop noise
        continue
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```
- `RETR_EXTERNAL` = outer outlines only. `RETR_TREE` = full parent/child hierarchy (holes inside shapes).
- `CHAIN_APPROX_SIMPLE` stores only the corner points, which saves memory.
- **Shape type:** `approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)` → 4 points means a rectangle.
- **Rotated box:** `minAreaRect`. **Centroid:** `M = cv2.moments(c); cx = M["m10"] / M["m00"]`.
- In OpenCV 4, `findContours` returns **2 values**; OpenCV 3 returned 3.
🎯 *"Binarize so objects are white, findContours, filter by area, then use boundingRect or approxPolyDP to get boxes and shapes."*

**2.9 Contours vs connected components?** ⭐
- **Contours** = outlines plus geometry (shape, hierarchy).
- **`connectedComponentsWithStats`** = labels every blob and gives its area, box and centroid in one fast call. Good for counting or filtering blobs.
```python
n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)   # stats[i] = x, y, w, h, area
```

**2.10 Hough transform: lines and circles?** ⭐⭐⭐
**The idea:** every edge pixel **votes** for all the lines (or circles) that could pass through it. Shapes that collect many votes are detected. A line is parameterised as `ρ = x·cosθ + y·sinθ`.
```python
lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 180, threshold=80,
                        minLineLength=50, maxLineGap=10)            # segments x1, y1, x2, y2

g = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=30, minRadius=10, maxRadius=60)
if circles is not None:
    for x, y, r in np.round(circles[0]).astype(int):
        cv2.circle(img, (x, y), r, (0, 0, 255), 2)
```
**HoughCircles parameters**
- `minDist`: minimum distance between circle centres.
- `param1`: the upper Canny threshold.
- `param2`: the **vote threshold**. Lower it to find more circles (and more false ones); raise it for fewer, more confident ones.
- **Tighten `minRadius` and `maxRadius`.** This is the biggest win for accuracy.
🎯 *"Hough lets edge pixels vote in parameter space; peaks are lines or circles. For circles I blur first and tune param2 and the radius range."*

**2.11 Template matching: how does it work and when does it fail?** ⭐⭐
It slides a small template over the image and scores the similarity at every position.
```python
res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)       # for TM_SQDIFF the best match is the MIN
if max_val > 0.8:
    th, tw = tmpl.shape[:2]
    cv2.rectangle(img, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 255, 0), 2)
```
**Multiple matches:** `np.where(res >= 0.8)` followed by NMS.
⚠️ It fails when the target is a different **scale, rotation** or appearance. Then use multi-scale matching or feature matching (ORB/SIFT).
🎯 *"Template matching slides a patch and scores similarity. It's fast but not scale- or rotation-invariant."*

**2.12 How do you detect changes between two images?** ⭐⭐⭐
1. **Align** the images first (feature matching + homography, or ECC). Otherwise a tiny shift makes everything look changed.
2. Diff and threshold, then clean the mask with morphology.
3. Find contours to get boxes around the changes.
```python
diff = cv2.absdiff(old_gray, new_gray)
_, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
mask = cv2.dilate(mask, None, iterations=2)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
**SSIM** (`skimage.metrics.structural_similarity(a, b, full=True)`) compares local structure, so it's more robust than a raw pixel diff to small brightness or noise differences.
🎯 *"Align first, then absdiff or SSIM, threshold, morphology, and contours to localise what changed."*

**2.13 Feature matching: ORB vs SIFT, and homography?** ⭐⭐⭐
- **Keypoint** = a distinctive point, such as a corner. **Descriptor** = a vector describing the patch around it.
- **SIFT:** 128 floats per point, very robust to scale and rotation, compared with L2 distance. Free since its patent expired in 2020.
- **ORB:** binary descriptor, **much faster**, free, compared with **Hamming** distance.
- **Homography:** a 3×3 matrix that maps one plane onto another. It needs **at least 4 point pairs**, and **RANSAC** throws out bad matches.
```python
orb = cv2.ORB_create(2000)
k1, d1 = orb.detectAndCompute(g1, None)
k2, d2 = orb.detectAndCompute(g2, None)
pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]   # Lowe ratio test
src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
aligned = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
```
**Ratio test:** keep a match only if the best match is clearly better than the second best.
🎯 *"Detect and describe keypoints, match them with a ratio test, then fit a homography with RANSAC to align the images."*

**2.14 Affine vs perspective transform?** ⭐⭐
| | Affine | Perspective (homography) |
|---|---|---|
| Matrix | 2×3 | 3×3 |
| Points needed | 3 | 4 |
| Keeps parallel lines parallel? | ✅ Yes (rotate, scale, shear, shift) | ❌ No: parallel lines can meet (camera tilt) |
| Example | Rotating an image | Scanning a document photographed at an angle |
```python
M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0);  rot = cv2.warpAffine(img, M, (w, h))
P = cv2.getPerspectiveTransform(np.float32(src4), np.float32(dst4));  flat = cv2.warpPerspective(img, P, (W, H))
```
🎯 *"Affine keeps parallel lines and needs 3 points; perspective handles camera tilt and needs 4."*

**2.15 Histogram equalization vs CLAHE?** ⭐⭐
- **equalizeHist:** spreads brightness across the whole image (global). It can blow out some regions and amplify noise.
- **CLAHE:** equalizes in small tiles, with a **clip limit** so noise isn't over-amplified. Better for uneven lighting.
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8));  out = clahe.apply(gray)
```
🎯 *"CLAHE is local, contrast-limited equalization. It boosts detail without amplifying noise much."*

**2.16 Reading video and background subtraction?** ⭐⭐
```python
cap = cv2.VideoCapture("v.mp4")          # 0 = webcam
bg = cv2.createBackgroundSubtractorMOG2()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fg = bg.apply(frame)                 # moving pixels = white
cap.release()
```
🎯 *"MOG2 learns a per-pixel background model; whatever doesn't fit it is foreground, i.e. motion."*

**2.17 Classical OpenCV vs deep learning: when do you use which?** ⭐⭐⭐
| Classical (OpenCV) | Deep learning |
|---|---|
| Controlled input (scans, drawings, a fixed camera) | Varied real-world scenes (lighting, pose, clutter) |
| Little or no labelled data | Lots of labelled data available |
| Needs to be explainable, fast, CPU-only | Accuracy matters more than simplicity |
In practice you **combine** them: classical steps for pre/post-processing, DL for the hard recognition part.
🎯 *"Classical methods when the conditions are controlled and data is scarce; deep learning when appearance varies. Often a hybrid."*

---

## 3. Deep learning for vision

**3.1 Why CNNs instead of fully connected networks for images?** ⭐⭐⭐
- **Local connectivity:** each filter looks at a small patch.
- **Weight sharing:** the same filter is reused everywhere in the image.
- Together these mean **far fewer parameters**, and a pattern is recognised wherever it appears.
- Layers build up a hierarchy: **edges → textures → parts → objects**.
🎯 *"Convolutions share weights over local patches, so CNNs are parameter-efficient and translation-equivariant, and they learn features from edges up to objects."*

**3.2 Conv output size and parameter count?** ⭐⭐⭐
- **Output size** = `(W − K + 2P) / S + 1`. Example: 224, K=3, P=1, S=1 → 224. A 2×2 pool with stride 2 halves the size.
- **Parameters** = `K·K·C_in·C_out + C_out` (the last term is the biases). Example: 3×3, 64→128 = **73,856**.
🎯 *"Output = (W − K + 2P)/S + 1; parameters = K²·Cin·Cout + Cout."*

**3.3 What do pooling, stride, padding and receptive field mean?** ⭐⭐
- **Pooling:** shrinks the feature map and gives a little shift-invariance (max-pool keeps the strongest response).
- **Stride:** the step size; stride 2 halves the size.
- **Padding:** adds a border so the size is kept ("same" padding).
- **Receptive field:** how much of the input one output pixel "sees". Two 3×3 convs = 5×5; three 3×3 = 7×7.
🎯 *"The receptive field grows with depth; stacking 3×3 convs reaches a large field cheaply."*

**3.4 Why does VGG use stacked 3×3 convs? What is a 1×1 conv?** ⭐⭐
- Three 3×3 layers see the same 7×7 area as one 7×7 layer, with **fewer parameters** (27C² vs 49C²) and **more non-linearities**.
- A **1×1 conv** mixes channels and changes their number cheaply (a bottleneck), as in ResNet and Inception.
🎯 *"Stacked 3×3s give a large receptive field with fewer parameters; 1×1 convs change the channel count cheaply."*

**3.5 ResNet skip connections: why do they work?** ⭐⭐⭐
`output = F(x) + x`. Each block only learns the **change** (residual), and gradients flow straight back through the `+ x` path. Without this, very deep networks get *worse* (vanishing gradients and the degradation problem).
🎯 *"Skip connections give gradients a shortcut, so networks with 100+ layers still train."*

**3.6 BatchNorm and Dropout: what are they, and how do they behave at inference?** ⭐⭐⭐
- **BatchNorm:** normalises activations per batch → faster, more stable training and tolerates higher learning rates. **At inference it uses running mean/variance.**
- **Dropout:** randomly zeroes neurons during training → less overfitting. **Off at inference.**
- That's why you must call `model.eval()`, and use `torch.no_grad()` to save memory.
```python
model.eval()
with torch.no_grad():
    out = model(x)
```
🎯 *"BatchNorm stabilises training, Dropout regularises; model.eval() switches both to inference behaviour."*

**3.7 Which activation and which loss?** ⭐⭐⭐
| Task | Last activation | Loss |
|---|---|---|
| Multi-class (one label) | Softmax | Cross-entropy |
| Binary / multi-label | Sigmoid | Binary cross-entropy |
| Box regression | none | L1 / Smooth-L1 / **IoU-based (GIoU, CIoU)** |
| Segmentation | Sigmoid or softmax per pixel | CE + **Dice** |
| Heavy class imbalance | | **Focal loss** (down-weights easy examples) |

Hidden layers: **ReLU** (or SiLU/GELU), which avoids the saturation of sigmoid.
🎯 *"Softmax + CE for single-label, sigmoid + BCE for multi-label, IoU losses for boxes, Dice for segmentation masks."*

**3.8 Overfitting vs underfitting: how do you fix each?** ⭐⭐⭐
- **Overfitting** (train high, validation low): more data or **augmentation**, weight decay, dropout, early stopping, a smaller model, pretrained weights.
- **Underfitting** (both low): a bigger model, train longer, less regularisation, a better learning rate.
🎯 *"Read the train-vs-validation gap: a gap means overfitting, both low means underfitting."*

**3.9 Transfer learning: how do you do it?** ⭐⭐⭐
Start from ImageNet-pretrained weights, which already "know" edges and shapes.
- **Small data:** freeze the backbone and train only the new head.
- **More data:** fine-tune everything with a **low learning rate**.
- Use the **same preprocessing** the pretrained model used (RGB, ImageNet mean/std).
```python
m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
for p in m.parameters():
    p.requires_grad = False
m.fc = nn.Linear(m.fc.in_features, num_classes)   # only this layer trains
# normalisation: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
```
🎯 *"Reuse pretrained features: freeze and train the head on small data, fine-tune with a low learning rate on more data."*

**3.10 Data augmentation: what and why?** ⭐⭐⭐
Flips, rotation, scale and crop, colour jitter, blur and noise; YOLO adds **mosaic** and **mixup**. They teach invariance and reduce overfitting.
⚠️ **Labels must stay valid:** transform the boxes and masks too, and don't flip digits or text (a flipped 6 can look like a 9).
🎯 *"Augmentation creates realistic variations so the model generalises. The geometric ones must transform the labels too."*

**3.11 Optimizers and learning rate?** ⭐⭐
- **SGD + momentum:** often generalises better; the default in YOLO.
- **Adam / AdamW:** converges faster; the default for transformers.
- The **learning rate** is the most important hyperparameter. Use a **warmup** then a **cosine or step decay**.
🎯 *"SGD with momentum for CNN detectors, AdamW for transformers, always with warmup and LR decay."*

**3.12 Class imbalance: how do you handle it?** ⭐⭐
Use a weighted loss or **focal loss**, oversample or augment the rare class, and judge with **precision, recall, F1 or PR-AUC**, not accuracy.
🎯 *"99% accuracy can mean the model ignores the rare class, so I weight the loss and judge by F1 or PR-AUC."*

**3.13 Model trains badly: how do you debug it?** ⭐⭐
1. **Visualise the data and labels.** Most bugs are wrong boxes, BGR/RGB mix-ups or bad normalisation.
2. Check that it can **overfit a tiny batch**. If it can't, there's a bug in the model or loss.
3. Check the learning rate (loss exploding → too high; flat → too low).
4. Check for data leakage (near-duplicate frames in both train and test).
5. If it's good in validation but bad in production → **domain shift**. Collect real data.
🎯 *"I check the data first, overfit one batch, then tune the learning rate; production failures are usually domain shift."*

---

## 4. Object detection

**4.1 Classification vs detection vs segmentation?** ⭐⭐⭐
| Task | Output |
|---|---|
| Classification | One label for the whole image |
| Detection | **Boxes + labels** for each object |
| Semantic segmentation | A class for **every pixel** (all cars are just "car") |
| Instance segmentation | A **separate mask per object** (car 1, car 2) |
| Panoptic | Semantic + instance together |

**4.2 IoU: what is it? Write it.** ⭐⭐⭐
IoU = **overlap area ÷ union area**. 1 = perfect match, 0 = no overlap. It's used for matching predictions to ground truth, for NMS, for metrics and for losses.
```python
def iou(a, b):                                   # boxes as [x1, y1, x2, y2]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + 1e-9)
```
🎯 *"IoU is intersection over union; above 0.5 usually counts as a correct detection."*

**4.3 NMS: what is it? Write it.** ⭐⭐⭐
Detectors predict **several overlapping boxes for the same object**. NMS keeps the highest-scoring box and removes the others that overlap it heavily, then repeats.
```python
def nms(boxes, scores, thr=0.5):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i, rest = order[0], order[1:]
        keep.append(i)
        order = rest[np.array([iou(boxes[i], boxes[j]) < thr for j in rest], dtype=bool)]
    return keep
```
- The **confidence threshold** drops weak boxes; the **NMS IoU threshold** decides what counts as a duplicate. They are different settings.
- **Soft-NMS** lowers the scores of overlapping boxes instead of deleting them, which helps with crowded objects.
🎯 *"NMS keeps the best box and suppresses overlapping duplicates above an IoU threshold."*

**4.4 One-stage vs two-stage detectors?** ⭐⭐⭐
| | Two-stage (Faster R-CNN) | One-stage (YOLO, SSD, RetinaNet) |
|---|---|---|
| How | ① the RPN proposes regions → ② each region is classified and its box refined | Predicts boxes + classes in **one pass** |
| Speed | Slower | **Real-time** |
| Accuracy | Traditionally higher, especially on small objects | Now very close |
🎯 *"Two-stage proposes regions then refines them, which is more accurate but slower; one-stage predicts everything in one pass, which is fast."*

**4.5 How does YOLO work, simply?** ⭐⭐⭐
- The image is divided into **grids at 3 scales** (fine for small objects, coarse for large).
- Each cell predicts boxes + an **objectness score** (is there an object?) + **class probabilities**.
- **Backbone** extracts features → **neck (FPN/PAN)** mixes the scales → **head** predicts → **NMS** cleans up.
- **Anchor-based** versions (v3–v7) refine preset box shapes. **Anchor-free** versions (v8+) predict the box directly from a point.
🎯 *"YOLO looks once: a CNN predicts boxes, objectness and classes on multi-scale grids, then NMS removes duplicates."*

**4.6 Anchors vs anchor-free?** ⭐⭐
- **Anchors:** preset box shapes per cell; the model predicts offsets from them. They must suit the dataset (k-means on your boxes).
- **Anchor-free:** predicts the centre plus distances to the box edges. There are fewer settings to tune, and it's the newer default.
🎯 *"Anchors are prior box shapes the model adjusts; anchor-free detectors predict boxes directly, so there are fewer hyperparameters."*

**4.7 What is FPN / multi-scale features?** ⭐⭐
- Deep layers know **what** something is but have coarse positions; shallow layers know **where** but have weak semantics.
- An **FPN** adds deep features back into shallow ones (top-down + lateral connections), so every scale has both.
🎯 *"FPN combines deep semantic features with high-resolution shallow ones, so objects of all sizes are detected well."*

**4.8 Precision, recall, F1, AP, mAP?** ⭐⭐⭐
- **Precision** = TP / (TP + FP): *"When it says object, is it right?"* (few false alarms)
- **Recall** = TP / (TP + FN): *"Did it find them all?"* (few misses)
- **F1** = 2PR / (P + R)
- **AP** = the area under the precision–recall curve for one class.
- **mAP@0.5** = AP averaged over classes at IoU 0.5. **mAP@0.5:0.95** (COCO) = also averaged over 10 IoU thresholds, so it rewards tight boxes.
- A **TP** = the right class **and** IoU above the threshold; each ground-truth box can be matched only once.
- **Lower the confidence threshold** → higher recall, lower precision.
🎯 *"Precision is about false alarms, recall about misses; mAP@0.5:0.95 is the standard detection metric."*

**4.9 Annotation formats?** ⭐⭐⭐
| Format | Box | Units |
|---|---|---|
| **YOLO** (.txt) | `class cx cy w h` | **normalised 0–1** |
| **COCO** (.json) | `[x_min, y_min, w, h]` | pixels |
| **Pascal VOC** (.xml) | `xmin ymin xmax ymax` | pixels |
🎯 *"YOLO uses a normalised centre and size, COCO uses top-left plus width and height in pixels, VOC uses corners."*

**4.10 How do you improve small-object detection?** ⭐⭐
A higher input resolution, **tiling or slicing** (for example SAHI), an extra high-resolution head (P2), anchors that fit small objects, more small-object data, and less aggressive downsampling.
🎯 *"Give small objects more pixels: higher resolution, tiling, a finer detection head."*

**4.11 New object, little data: how do you build a detector?** ⭐⭐⭐
Start from a **pretrained YOLO** → label a few hundred varied images → augment → fine-tune → check the false positives and negatives → add **hard examples** → iterate.
🎯 *"Fine-tune a pretrained detector on a small, carefully labelled set, then iterate on its failure cases."*

---

## 5. Segmentation

**5.1 U-Net: how does it work?** ⭐⭐⭐
An **encoder** shrinks the image to learn *what* is there; a **decoder** grows it back to full size; **skip connections** copy fine detail from the encoder to the decoder to give sharp boundaries. It works well with small datasets (it's the classic medical-imaging model).
🎯 *"U-Net is an encoder-decoder with skip connections, so it gets both context and sharp boundaries."*

**5.2 Mask R-CNN?** ⭐⭐
Faster R-CNN + an extra **mask branch** that predicts a mask for each box. **RoIAlign** (no rounding of coordinates) replaces RoIPool, so the masks line up precisely.
🎯 *"Mask R-CNN adds a per-box mask head to Faster R-CNN; RoIAlign keeps the masks pixel-accurate."*

**5.3 Segmentation metrics: IoU vs Dice?** ⭐⭐
- **IoU** = |A∩B| / |A∪B|. **Dice** = 2|A∩B| / (|A| + |B|), so `Dice = 2·IoU / (1 + IoU)`.
- **Dice loss** handles small foreground objects (class imbalance) well.
🎯 *"Both measure mask overlap; Dice weights the overlap more and works well as a loss for imbalanced masks."*

**5.4 SAM (Segment Anything): how does it work?** ⭐⭐⭐
- **Promptable segmentation:** give it a **point, a box or a rough mask** and it returns the mask of that object.
- **Heavy ViT image encoder** (run **once** per image) + **prompt encoder** + **light mask decoder** (fast, so you can prompt many times).
- Trained on SA-1B (about 1.1 billion masks), so it works **zero-shot** on new domains. It gives **masks, not class names**.
- Ambiguous prompts → it returns **3 candidate masks** with quality scores.
🎯 *"SAM encodes the image once with a ViT, then a light decoder turns any point or box prompt into a mask, zero-shot."*

---

## 6. Embeddings and similarity

**6.1 What is an image embedding?** ⭐⭐⭐
A vector from the network's last hidden layer (for example VGG16 FC7 = 4096 numbers) that summarises **what the image looks like**. Similar images → nearby vectors. Used for search, matching, deduplication and re-identification.
🎯 *"An embedding is the network's compressed description of an image; similar images give nearby vectors."*

**6.2 Cosine similarity vs Euclidean distance?** ⭐⭐⭐
- **Cosine** = `a·b / (|a||b|)` compares **direction** (the pattern) and ignores magnitude (for example overall brightness or activation scale).
- After **L2-normalising** the vectors, cosine = dot product, and it ranks results the same way as Euclidean distance.
```python
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
```
For large-scale search, use a vector index (**FAISS**).
🎯 *"Cosine compares direction, not size, which is why it's the standard for comparing embeddings."*

**6.3 Siamese / triplet loss / CLIP: one line each?** ⭐
- **Siamese:** two inputs, shared weights, learns "same or different".
- **Triplet loss:** pulls anchor and positive together, pushes the negative away by a margin.
- **CLIP:** images and text in **one shared vector space**, so you can search images with text and classify zero-shot.

**6.4 ViT vs CNN?** ⭐⭐
- **ViT:** splits the image into 16×16 **patches** (224 → 196 tokens) → **self-attention** across all patches, so it has **global context** from the first layer.
- **CNN:** local filters with built-in assumptions (locality), so it learns well from less data.
- **ViT needs more data or pretraining** but scales better.
🎯 *"ViTs treat patches as tokens with global attention; CNNs have stronger built-in assumptions, so they're more data-efficient."*

---

## 7. OCR

**7.1 How do you get good OCR from Tesseract?** ⭐⭐⭐
Most of the accuracy comes from **preprocessing**:
- Grayscale → **upscale small text** → denoise → Otsu/adaptive threshold → **deskew**.
- Tesseract prefers **dark text on a light background**.
- Pick the right **page segmentation mode (PSM)**, then clean the output with **regex and confidence filtering**.
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
text = pytesseract.image_to_string(bw, config="--oem 3 --psm 6")
data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT)   # words + boxes + confidence
```
PSM: `6` = one block of text · `7` = a single line · `11` = sparse text. For digits only: `-c tessedit_char_whitelist=0123456789`.
🎯 *"OCR quality is mostly preprocessing: upscale, binarize, deskew, choose the right PSM, then validate with regex and confidence."*

**7.2 Tesseract vs deep-learning OCR?** ⭐⭐
- **Tesseract:** CPU, offline, good on clean printed text.
- **EasyOCR / PaddleOCR:** a **text detector + a recogniser (CRNN + CTC)**, much better on scene text, rotated text or noisy text.
🎯 *"Tesseract for clean documents offline; DL OCR for messy or rotated text."*

---

## 8. Video, tracking and camera

**8.1 How does multi-object tracking work?** ⭐⭐
**Tracking-by-detection:** detect objects in every frame, then **link** detections across frames.
- **SORT:** a Kalman filter predicts each track's next position, and the Hungarian algorithm matches predictions to detections by IoU.
- **DeepSORT:** adds **appearance embeddings**, so IDs survive occlusion.
- **ByteTrack:** also uses low-confidence detections, so fewer tracks are lost.
🎯 *"Detect every frame, predict motion with a Kalman filter, match by IoU and appearance to keep IDs."*

**8.2 Optical flow?** ⭐
It estimates **how pixels move** between frames, assuming brightness stays constant and motion is small.
- **Lucas-Kanade** (`calcOpticalFlowPyrLK`) = sparse: tracks selected points.
- **Farneback** = dense: computes motion for every pixel.

**8.3 Camera calibration: what and why?** ⭐⭐
- **Intrinsics** K: `fx, fy` (focal length), `cx, cy` (image centre). **Extrinsics:** R, t (where the camera is).
- **Distortion:** lens bending (barrel or pincushion).
- **Steps:** photograph a checkerboard → `findChessboardCorners` → `calibrateCamera` → `undistort`. Needed for measuring real sizes and for 3D.
🎯 *"Calibration finds the focal length, image centre and lens distortion, so pixels map correctly to the real world."*

---

## 9. Deployment

**9.1 How do you make a model faster?** ⭐⭐⭐
- A smaller model or input size.
- **FP16 / INT8 quantization** (INT8 needs calibration data).
- Export to **ONNX** → run with **TensorRT** (NVIDIA) or OpenVINO (Intel CPU).
- Batching, and keeping data on the GPU (avoid copying back and forth to the CPU).
- Skip frames in video.
```python
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=17)
```
🎯 *"A smaller model, lower precision, an optimised runtime like ONNX to TensorRT, and fewer CPU-GPU copies."*

**9.2 Latency vs throughput?** ⭐
- **Latency** = the time for **one** image (what matters for real-time).
- **Throughput** = images **per second** (batching raises it but can raise latency).

**9.3 Classic production bug?** ⭐⭐⭐
**Preprocessing mismatch** between training and inference: BGR vs RGB, a different resize or letterbox, missing normalisation, or forgetting `model.eval()`.
🎯 *"The first thing I check is that inference preprocessing exactly matches training."*

---

## 10. Coding-round favourites (practise writing these)
1. **IoU** → 4.2
2. **NMS** → 4.3
3. **Count objects / coins:** blur → Otsu (INV) → opening → `findContours` → filter by area → count
4. **Detect blur:** variance of the Laplacian → 2.7
5. **Find a shape:** contours + `approxPolyDP` → 2.8
6. **Colour object mask:** HSV + `inRange` → 1.3
7. **Align two images:** ORB + ratio test + homography → 2.13
8. **Diff two images:** `absdiff` → threshold → contours → 2.12
9. **Resize keeping aspect ratio / letterbox** → 1.4
10. **Convolution in NumPy** (2 loops: multiply the window by the kernel and sum)

---

## 11. Rapid fire (cover the answers and test yourself)
| Question | Answer |
|---|---|
| `imread` with a bad path? | Returns `None`, no error |
| OpenCV colour order? | BGR |
| Pixel indexing? | `img[y, x]` (row, column) |
| `cv2.resize` size order? | `(width, height)` |
| Interpolation for shrinking? | `INTER_AREA` |
| Interpolation for masks? | `INTER_NEAREST` |
| Remove salt-and-pepper noise? | Median blur |
| Blur but keep edges? | Bilateral |
| Remove small white specks? | Opening |
| Fill small holes? | Closing |
| Thin 1-pixel edges? | Canny |
| Why `CV_64F` in Sobel? | Keeps negative gradients |
| Quick blur detector? | Variance of the Laplacian |
| Threshold under uneven lighting? | Adaptive |
| Automatic threshold? | Otsu |
| Points for an affine / homography? | 3 / 4 (use RANSAC with more) |
| ORB distance? | Hamming |
| SIFT descriptor size? | 128 floats |
| OpenCV hue range? | 0–179 (red wraps, so two ranges) |
| `findContours` input? | Binary, white object on black |
| Template matching weakness? | Not scale- or rotation-invariant |
| HoughCircles: more circles? | Lower `param2` |
| What is a 1×1 conv for? | Changing the channel count cheaply |
| Dropout at inference? | Off (`model.eval()`) |
| BatchNorm at inference? | Uses running mean/variance |
| NMS removes? | Duplicate boxes for the same object |
| mAP@0.5:0.95? | Average over 10 IoU thresholds |
| Dice from IoU? | 2·IoU / (1 + IoU) |
| ViT tokens for 224 with 16×16 patches? | 196 (+1 CLS) |
| YOLO label format? | `class cx cy w h`, normalised |
| ImageNet mean? | [0.485, 0.456, 0.406] |

---

### If you only have 1 hour
Do sections **2 (all of OpenCV)**, **4.2–4.5 and 4.8**, **3.1, 3.5, 3.6, 3.8, 3.9**, **5.4**, **6.2**, **7.1**, then the **rapid fire** table.
