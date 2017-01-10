import os
import ROOT 
ROOT.gSystem.Load("libDelphes.so")
from ROOT import Tower
from ROOT import Muon
from ROOT import Electron
from ROOT import Track
import numpy
import math

def DRsq(p, q):
    # compute isolation in a cone of 0.3
    DeltaEta = p['Eta'] - q['Eta']
    DeltaPhi = p['Phi'] = q['Phi']
    # force deltaPhi in [-pi, pi]
    tooLarge = -2.*math.pi*(DeltaPhi > math.pi)
    tooSmall = 2.*math.pi*(DeltaPhi < -math.pi)
    DeltaPhi = DeltaPhi + tooLarge + tooSmall    
    return DeltaEta*DeltaEta+DeltaPhi*DeltaPhi

def Closest(p, tracks):
    iClosest = -99
    distance = 99999999.
    for i in range(len(tracks)):
        this_d = DRsq(p, tracks[i])
        if this_d < distance:
            distance = this_d
            iClosest = i
    return iClosest

def DeltaRsq(p, Eta, Phi):
    # compute isolation in a cone of 0.3
    DeltaEta = Eta - p['Eta']
    DeltaPhi = Phi - p['Phi']
    # force deltaPhi in [-pi, pi]
    tooLarge = -2.*math.pi*(DeltaPhi > math.pi)
    tooSmall = 2.*math.pi*(DeltaPhi < -math.pi)
    DeltaPhi = DeltaPhi + tooLarge + tooSmall    
    return DeltaEta*DeltaEta+DeltaPhi*DeltaPhi

def Iso(p, trkPt, trkEta, trkPhi):
    DRsq = DeltaRsq(p, trkEta, trkPhi)
    CloseTracks = DRsq < 0.3*0.3
    return trkPt[CloseTracks].sum()/p['PT']
            
def Convert():
    
    fileIN = ROOT.TFile.Open("/afs/cern.ch/user/m/mpierini/public/DANILO/ttbar_lepFilter_13TeV_994.root")
    tree = fileIN.Get("Delphes")

    for evt in tree:
        
        # Look for Muons
        myMuons = []
        nMu = len(evt.MuonTight)
        muEta = numpy.zeros((nMu, 1))
        muPhi = numpy.zeros((nMu, 1))
        muPt = numpy.zeros((nMu, 1))
        i = 0
        for mu in evt.MuonTight:
            myMu = ROOT.TLorentzVector()
            myMu.SetPtEtaPhiM(mu.PT, mu.Eta, mu.Phi, 0.)
            myMuons.append({'PT':myMu.Pt(), 'Eta': myMu.Eta(), 'Phi': myMu.Phi(), 'Px': myMu.Px(), 'Py': myMu.Py(), 'Pz': myMu.P(), \
                                'X': 0., 'Y': 0., 'Z': 0., \
                                'Dxy': 0., 'Charge': mu.Charge, 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
            muEta[i] = (myMu.Eta()) 
            muPhi[i] = (myMu.Phi())
            muPt[i] = (myMu.Pt())
            i = i +1 

        # Look for Electrons
        myElectrons = []
        nEle = len(evt.Electron)
        eleEta = numpy.zeros((nEle, 1))
        elePhi = numpy.zeros((nEle, 1))
        elePt = numpy.zeros((nEle, 1))
        i = 0
        for ele in evt.Electron:
            myEle = ROOT.TLorentzVector()
            myEle.SetPtEtaPhiM(ele.PT, ele.Eta, ele.Phi, 0.)
            myElectrons.append({'PT': myEle.Pt(), 'Eta': myEle.Eta(), 'Phi': myEle.Phi(), 'Px': myEle.Px(), 'Py': myEle.Py(), 'Pz': myEle.P(), \
                                'X': 0., 'Y': 0., 'Z': 0., \
                                'Dxy': 0., 'Charge': ele.Charge, 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
            eleEta[i] = (myEle.Eta()) 
            elePhi[i] = (myEle.Phi())
            elePt[i] = (myEle.Pt())
            i =i +1

        # loop over tracks
        #for i in range(EFlowTrack_size):
        #    print EFlowTrack.PT[i]

        # Tracks (excluding electrons and muons)
        nTracks = len(evt.EFlowTrack)
        trkPt = numpy.zeros((nTracks, 1))
        trkEta = numpy.zeros((nTracks, 1))
        trkPhi = numpy.zeros((nTracks, 1))
        myChargedHadrons = []
        myMuAsTrk = []
        myEleAsTrk = []
        i = 0
        for trk in evt.EFlowTrack:
            p = ROOT.TLorentzVector()
            p.SetPtEtaPhiM(trk.PT, trk.Eta, trk.Phi, 0.)
            myTrk = {'PT': p.Pt(), 'Eta': p.Eta(), 'Phi': p.Phi(), 'Px': p.Px(), 'Py': p.Py(), 'Pz': p.P(), \
                         'X': trk.X, 'Y': trk.Y, 'Z': trk.Z, \
                         'Dxy': trk.Dxy, 'Charge': trk.Charge, 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.}
            # ignore tracks identified as muons (through angular matching)
            if numpy.amin(DeltaRsq(myTrk, eleEta, elePhi)) < 1.E-3: 
                myEleAsTrk.append(myTrk)
                continue
            if numpy.amin(DeltaRsq(myTrk, muEta, muPhi)) < 1.E-3: 
                myMuAsTrk.append(myTrk)
                continue
            myChargedHadrons.append(myTrk)
            # for isolation
            trkEta[i] = trk.PT
            trkPhi[i] = trk.Eta
            trkPt[i] = trk.Phi
            i = i +1

        # for each electron, find its view as EFlowTrack and retreive X, Y, Z, and Dxy
        for ele in myElectrons:
            idx = Closest(ele, myEleAsTrk)
            if idx < 0: continue
            ele['Dxy'] = myEleAsTrk[idx]['Dxy']
            ele['X'] = myEleAsTrk[idx]['X']
            ele['Y'] = myEleAsTrk[idx]['Y']
            ele['Z'] = myEleAsTrk[idx]['Z']
        # for each muon, find its view as EFlowTrack and retreive X, Y, Z, and Dxy
        for mu in myMuons:
            idx = Closest(mu, myMuAsTrk)
            if idx < 0: continue
            mu['Dxy'] = myMuAsTrk[idx]['Dxy']
            mu['X'] = myMuAsTrk[idx]['X']
            mu['Y'] = myMuAsTrk[idx]['Y']
            mu['Z'] = myMuAsTrk[idx]['Z']
                
        myPhotons = []
        nGamma = len(evt.EFlowPhoton)
        gammaEta = numpy.zeros((nGamma, 1))
        gammaPhi = numpy.zeros((nGamma, 1))
        gammaPt = numpy.zeros((nGamma, 1))
        i = 0
        for gamma in evt.EFlowPhoton:
            p = ROOT.TLorentzVector()
            p.SetPtEtaPhiM(gamma.ET, gamma.Eta, gamma.Phi, 0.)
            myPhotons.append({'PT': p.Pt(), 'Eta': p.Eta(), 'Phi': p.Phi(), 'Px': p.Px(), 'Py': p.Py(), 'Pz': p.P(), \
                                  'X': 0., 'Y': 0., 'Z': 0., \
                                  'Dxy': 0., 'Charge': 0., 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
            gammaEta[i] = p.Eta()
            gammaPhi[i] = p.Phi()
            gammaPt[i] = p.Pt()
            i = i +1

        # neutral hadrons
        myNeutralHadrons = []
        nNeuHad = len(evt.EFlowNeutralHadron)
        neuEta = numpy.zeros((nNeuHad, 1))
        neuPhi = numpy.zeros((nNeuHad, 1))
        neuPt = numpy.zeros((nNeuHad, 1))
        i = 0
        for NeuHad in evt.EFlowNeutralHadron:
            p = ROOT.TLorentzVector()
            p.SetPtEtaPhiM(NeuHad.ET, NeuHad.Eta, NeuHad.Phi, 0.)
            myNeutralHadrons.append({'PT': p.Pt(), 'Eta': p.Eta(), 'Phi': p.Phi(), 'Px': p.Px(), 'Py': p.Py(), 'Pz': p.P(), \
                                         'X': 0., 'Y': 0., 'Z': 0., \
                                         'Dxy': 0., 'Charge': 0., 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
            neuEta[i] = NeuHad.Eta
            neuPhi[i] = NeuHad.Phi
            neuPt[i] = NeuHad.ET
            i = i +1

        # compute isolation
        for mu in myMuons:
            mu['MuIso'] = Iso(mu, muPt, muEta, muPhi) -1.
            mu['EleIso'] = Iso(mu, elePt, eleEta, elePhi)
            mu['ChHadIso'] = Iso(mu, trkPt, trkEta, trkPhi)
            mu['NeuHadIso'] = Iso(mu, neuPt, neuEta, neuPhi)
            mu['GammaIso'] = Iso(mu, gammaPt, gammaEta, gammaPhi)
            
        for ele in myElectrons:
            ele['MuIso'] = Iso(ele, muPt, muEta, muPhi)
            ele['EleIso'] = Iso(ele, elePt, eleEta, elePhi) -1.
            ele['ChHadIso'] = Iso(ele, trkPt, trkEta, trkPhi)
            ele['NeuHadIso'] = Iso(ele, neuPt, neuEta, neuPhi)
            ele['GammaIso'] = Iso(ele, gammaPt, gammaEta, gammaPhi)

        for gamma in myPhotons:
            gamma['EleIso'] = Iso(gamma, elePt, eleEta, elePhi) 
            gamma['MuIso'] = Iso(gamma, muPt, muEta, muPhi)
            gamma['ChHadIso'] = Iso(gamma, trkPt, trkEta, trkPhi)
            gamma['NeuHadIso'] = Iso(gamma, neuPt, neuEta, neuPhi)
            gamma['GammaIso'] = Iso(gamma, gammaPt, gammaEta, gammaPhi) -1

        for p in myNeutralHadrons:
            p['EleIso'] = Iso(p, elePt, eleEta, elePhi)
            p['MuIso'] = Iso(p, muPt, muEta, muPhi)
            p['ChHadIso'] = Iso(p, trkPt, trkEta, trkPhi)
            p['NeuHadIso'] = Iso(p, neuPt, neuEta, neuPhi) -1
            p['GammaIso'] = Iso(p, gammaPt, gammaEta, gammaPhi) 

        

if __name__ == "__main__":
    Convert()
    
