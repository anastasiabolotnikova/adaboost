import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def get_grayscale_fovea(file):
    fovea_grey = []
    f=open(file)
    line=f.readline()
    while line != "grayscale\n":
        line=f.readline()
    line=f.readline()
    while line!="Edges\n":
        fovea_grey.append([int(el) for el in line.split()])
        line=f.readline()
    f.close()
    return fovea_grey


def onclick(event):
  print(event.x)


def display(fovea, output, filename):
    ax = plt.gca()
    ax1 = plt.gca()
    fig = plt.gcf()
    implot = ax.imshow(fovea,  interpolation='nearest')
    ax1 = fig.add_subplot(111)
    ball = []
    def onclick(event):
        if event.xdata != None and event.ydata != None:
            coordinate = (int(event.xdata), int(event.ydata))
            print(coordinate)
            output.write("("+str(coordinate)+")")

            ball.append(coordinate)

            if(len(ball) == 2):
              correctball = get_ordered_coords(ball)

              rect = patches.Rectangle(ball[0],   # (x,y)
                      ball[1][0]-ball[0][0],          # width
                      ball[1][1]-ball[0][1],          # height
                      alpha=1,
                      facecolor='none',
                      edgecolor="red"
              )
              ax1.add_patch(rect)
              fig.canvas.draw()
              
              def decide(event):
                if event.key == 'x':
                  saveball(fovea,correctball, filename)
                  plt.close()
                else:
                  ball = []
                  rect.remove()
                  fig.canvas.draw()
              fig.canvas.mpl_connect('key_press_event', decide)          


    def press(event):
      if event.key == 'x':
        plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    kid = fig.canvas.mpl_connect('key_press_event', press)
    plt.show()

def get_ordered_coords(ball):
  p1 = ball[0]
  p2 = ball[1]
  xstart = p1[0]
  xend = p2[0]
  ystart = p1[1]
  yend = p2[1]
  
  if(xstart > xend):
    xend, xstart = xstart, xend
  if(ystart > yend):
    yend, ystart = ystart, yend
  return ((xstart, ystart), (xend,yend))

def saveball(fovea, ball, filename):
  p1 = ball[0]
  p2 = ball[1]
  xstart = p1[0]
  xend = p2[0]
  ystart = p1[1]
  yend = p2[1]
  
  if(xstart > xend):
    xend, xstart = xstart, xend
  if(ystart > yend):
    yend, ystart = ystart, yend
  ballfovea = open("training+/"+filename, "w")

  for y in range(ystart+1, yend+1):
    for x in range(xstart+1, xend+1):
      ballfovea.write(str(fovea[y][x]))
      ballfovea.write(" ")
    ballfovea.write("\n")
  ballfovea.close()



path = "training0406_2/"

output = open('output','w')
files = os.listdir(path)
idx = 0
# plt.ion()
if(not os.path.isdir("training+")):
  os.mkdir("training+")

for file in os.listdir(path):
    if not file.endswith(".png"):
        print(file)
        output.write(file + " ")
        gray_fovea = get_grayscale_fovea(path + file)
        display(gray_fovea, output, file)
        output.write("\n")

output.close()


