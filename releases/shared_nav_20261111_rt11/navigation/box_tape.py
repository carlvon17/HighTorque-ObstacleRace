"""Register measured ground tape against a crate-anchored local tape map."""
import cv2
import numpy as np


class BoxTape:
    cell = .020
    center = 180

    def __init__(self):
        self.painted = np.zeros((420, 420), np.uint8)

    def match(self, points, position, anchored):
        """Return a correction only if tape is observable at this capture."""
        if len(points) < 40:
            return None, False
        cells = np.unique(np.rint((points+position)/self.cell).astype(np.int32), axis=0)
        cells += self.center
        inside = ((cells >= 6) & (cells < self.painted.shape[0]-6)).all(axis=1)
        cells = cells[inside]
        if len(cells) < 25:
            return None, False
        prior = cv2.distanceTransform(1-self.painted, cv2.DIST_L2, 3)
        x,y = cells.T
        available=prior[y,x]<4
        correction=None
        longitudinal=False
        if available.sum()>=25 and np.ptp(x[available])>=12:
            x,y=x[available],y[available]
            shifts=range(-3,4)
            scores=np.array([[np.minimum(prior[y+j,x+i],3.5).mean() for i in shifts] for j in shifts])
            iy,ix=np.unravel_index(np.argmin(scores),scores.shape)
            best=scores[iy,ix]
            if best<.85:
                # A straight boundary can recover side offset but cannot fix
                # travel along its length. A bend or transverse paint must
                # penalize both signs of a longitudinal perturbation.
                dy=iy-3
                y_good=abs(dy)<3 and scores[0,ix]-best>.19 and scores[-1,ix]-best>.19
                dx=ix-3
                longitudinal=abs(dx)<3 and scores[iy,0]-best>.060 and scores[iy,-1]-best>.060
                if y_good:
                    correction=self.cell*np.array([dx if longitudinal else 0.,dy])
        if anchored:
            self.painted[cells[:,1],cells[:,0]]=1
        return correction,longitudinal
