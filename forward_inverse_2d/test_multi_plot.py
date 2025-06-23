import pyvista as pv
import multiprocessing

def plot_sphere():
    plotter = pv.Plotter()
    plotter.add_mesh(pv.Sphere(), color='red')
    plotter.show()

def plot_cube():
    plotter = pv.Plotter()
    plotter.add_mesh(pv.Cube(), color='blue')
    plotter.show()

if __name__ == '__main__':
    p1 = multiprocessing.Process(target=plot_sphere)
    p2 = multiprocessing.Process(target=plot_cube)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
