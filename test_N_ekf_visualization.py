"""Test N: cinematic plots for moving cube, V4 RGB-D sensing, and CubeEKF.
Run from repository root: .\\.venv\\Scripts\\python.exe test_N_ekf_visualization.py
Cube truth is used ONLY to drive motion and evaluate errors.
The arm is held at its scan pose kinematically: this is an isolated
perception/EKF test, NOT a torque-controller or full-system test.
"""
import csv
from pathlib import Path
import numpy as np
import mujoco
from src.control.ik import make_ik
from src.perception.sensing import camera
from src.estimation.cube_ekf import CubeEKF

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'test_N_results'
OUT.mkdir(exist_ok=True)
MOVE_DURATION, TOTAL_DURATION, SPEED, PERIOD = 6., 10., .035, .10

def main():
    m = mujoco.MjModel.from_xml_path(str(ROOT/'robot_model'/'pick_scene.xml'))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0)
    j = m.joint('box_joint').id
    vadr = int(m.jnt_dofadr[j])
    bid = m.body('box').id
    q_scan = np.asarray(make_ik(m)(np.array([.27,.10,.30]), d.qpos[:6]),float)
    if q_scan.shape != (6,) or not np.all(np.isfinite(q_scan)):
        raise RuntimeError('Scan IK returned invalid joint angles')
    d.qpos[:6] = q_scan
    # Test N isolates camera sensing and cube EKF from the arm controller.
    # The six UR3e actuators are TORQUE motors, not position servos.
    # Hold the scan pose kinematically during this estimator-only test.
    d.qvel[:6] = 0.0
    d.ctrl[:6] = 0.0
    mujoco.mj_forward(m,d)
    cam = camera(m, W=640, H=480, seed=42)
    ekf = CubeEKF(dt=PERIOD, K=cam.K, top_z=.04)
    rows=[]
    next_camera=0.
    last_camera=None
    try:
        while d.time < TOTAL_DURATION:
            t=float(d.time)
            # Only the simulation motion driver sees the prescribed velocity.
            if t < MOVE_DURATION:
                vx,vy=SPEED*np.cos(2*t), SPEED*np.sin(2*t)
            else:
                vx,vy=0.,0.
            d.qvel[vadr:vadr+2]=[vx,vy]
            # No angle-as-torque command. Re-impose a fixed scan pose after
            # each physics step and recompute camera/body transforms.
            d.ctrl[:6] = 0.0
            mujoco.mj_step(m,d)
            d.qpos[:6] = q_scan
            d.qvel[:6] = 0.0
            mujoco.mj_forward(m,d)
            if d.time + 1e-9 < next_camera:
                continue
            now=float(d.time)
            if ekf.initialized and last_camera is not None:
                ekf.predict(now-last_camera)
            last_camera=now
            next_camera=now+PERIOD
            rgb,depth=cam.take_photo(d,noise_scale=1.0)
            result=cam.locate(rgb,depth,d.qpos[:6].copy())
            gt=d.xpos[bid,:2].copy()  # evaluation only
            if result is None:
                measured=np.array([np.nan,np.nan]); accepted=False; score=np.nan
            else:
                centre,det=result
                measured=centre[:2].copy()
                # V4's final geometry centre must be used for ALL measurement
                # coordinates. Its original det['z'] is a raw blob median and
                # is NOT necessarily the depth of the fitted centre.
                measurement=dict(det)
                measurement['z']=float(det['p_cam'][2])
                cam_pos,cam_R=cam.pose(d.qpos[:6].copy())
                accepted=bool(ekf.update(measurement,cam_pos,cam_R))
                score=float(det.get('fit_score_px2',np.nan))
            est=ekf.position[:2].copy() if ekf.initialized else np.array([np.nan,np.nan])
            speed=ekf.speed if ekf.initialized else np.nan
            raw_err=float(np.linalg.norm(measured-gt)*1000) if np.all(np.isfinite(measured)) else np.nan
            ekf_err=float(np.linalg.norm(est-gt)*1000) if np.all(np.isfinite(est)) else np.nan
            # Posterior covariance: 1-sigma world X/Y position uncertainty (metres).
            sigma_x = float(np.sqrt(max(0.0, ekf.P[0, 0]))) if ekf.initialized else np.nan
            sigma_y = float(np.sqrt(max(0.0, ekf.P[1, 1]))) if ekf.initialized else np.nan
            rows.append(dict(sigma_x=sigma_x,sigma_y=sigma_y,t=now,true_x=gt[0],true_y=gt[1],raw_x=measured[0],raw_y=measured[1],
                             ekf_x=est[0],ekf_y=est[1],raw_err_mm=raw_err,ekf_err_mm=ekf_err,
                             ekf_speed=speed,ekf_vx=(float(ekf.velocity[0]) if ekf.initialized else np.nan),ekf_vy=(float(ekf.velocity[1]) if ekf.initialized else np.nan),detected=int(result is not None),accepted=int(accepted),
                             fit_score_px2=score))
            if len(rows)%10==1:
                print(f't={now:5.2f}s V4={"OK" if result is not None else "MISS"} '
                      f'raw_err={raw_err:.3f}mm ekf_err={ekf_err:.3f}mm speed={speed:.4f}m/s')
    finally:
        cam.ren.close()
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('\n=== TEST N SUMMARY ===')
    print('Measurements:',sum(r['detected'] for r in rows),'/',len(rows),
          'EKF accepted:',sum(r['accepted'] for r in rows))
    for name,subset in [('moving',[r for r in rows if r['t']<MOVE_DURATION]),
                        ('stopped',[r for r in rows if r['t']>=MOVE_DURATION+1]),
                        ('all',rows)]:
        print('\n'+name)
        for label,key in [('raw V4','raw_err_mm'),('EKF','ekf_err_mm')]:
            vals=np.array([r[key] for r in subset],float); vals=vals[np.isfinite(vals)]
            print(f'  {label} RMSE [mm]: {np.sqrt(np.mean(vals**2)):.4f}' if vals.size else f'  {label}: no data')
        if name=='stopped':
            v=np.array([r['ekf_speed'] for r in subset],float)
            print(f'  EKF mean speed [m/s]: {np.nanmean(v):.6f}')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                             'axes.spines.right': False, 'figure.facecolor': 'white'})
        t=np.array([r['t'] for r in rows], dtype=float)
        def values(k): return np.array([r[k] for r in rows],dtype=float)
        # Six individually exportable 16:9 visuals for presentation editing.
        def setup(title, ylabel=None):
            fig,ax=plt.subplots(figsize=(12.8,7.2),dpi=150)
            ax.set_title(title,fontsize=19,weight='bold',pad=18)
            if ylabel: ax.set_ylabel(ylabel)
            ax.grid(alpha=.18)
            return fig,ax
        def save(fig,name):
            fig.tight_layout(pad=2.5)
            fig.savefig(OUT/name,dpi=170,facecolor='white')
            plt.close(fig)
        for key,title,name in [('x','World X position','01_X_position.png'),
                               ('y','World Y position','02_Y_position.png')]:
            fig,ax=setup(title,'Position (mm)')
            for prefix,label,style,width in [('true','Ground truth','-',3.2),
                                              ('raw','Raw RGB-D','.',1.3),
                                              ('ekf','EKF estimate','--',2.4)]:
                ax.plot(t,1000*values(prefix+'_'+key),style,label=label,
                        linewidth=width,markersize=5,alpha=.88)
            ax.axvline(MOVE_DURATION,ls=':',color='gray',label='Cube stops')
            ax.set_xlabel('Time (s)');ax.legend(loc='best');save(fig,name)
        fig,ax=setup('Top-down cube trajectory','World Y (mm)')
        ax.plot(values('true_x')*1000,values('true_y')*1000,label='Ground truth',lw=3)
        ax.scatter(values('raw_x')*1000,values('raw_y')*1000,label='Raw RGB-D',s=13,alpha=.55)
        ax.plot(values('ekf_x')*1000,values('ekf_y')*1000,'--',label='EKF estimate',lw=2.3)
        ax.set_xlabel('World X (mm)');ax.set_aspect('equal',adjustable='datalim')
        ax.legend();save(fig,'03_XY_trajectory.png')
        fig,ax=setup('Position estimation error','2D error (mm)')
        ax.plot(t,values('raw_err_mm'),label='Raw RGB-D',alpha=.75)
        ax.plot(t,values('ekf_err_mm'),label='EKF estimate',lw=2)
        ax.axvline(MOVE_DURATION,ls=':',color='gray',label='Cube stops')
        ax.set_xlabel('Time (s)');ax.legend();save(fig,'04_error_comparison.png')
        fig,ax=setup('Estimated speed and stop detection','Speed (mm/s)')
        ax.plot(t,values('ekf_speed')*1000,label='EKF estimated speed',lw=2.7)
        ax.plot(t,np.where(t<MOVE_DURATION,SPEED*1000,0),label='Ground-truth commanded speed',ls='--',lw=2)
        ax.axvline(MOVE_DURATION,ls=':',color='gray',label='Cube stops')
        ax.set_xlabel('Time (s)');ax.legend();save(fig,'05_speed_estimation.png')
        fig,ax=setup('EKF estimates motion for predictive tracking','World X (mm)')
        ax.plot(t,values('true_x')*1000,label='Ground truth',lw=3)
        ax.plot(t,values('ekf_x')*1000,label='EKF current position',lw=2)
        # Visualizes the constant-velocity lookahead, NOT measured robot TCP.
        # Reconstruct velocity from the filter by recording it at every update.
        ax.plot(t,(values('ekf_x')+0.15*values('ekf_vx'))*1000,
                label='0.15 s lookahead target',lw=2,ls='--')
        ax.axvline(MOVE_DURATION,ls=':',color='gray',label='Cube stops')
        ax.set_xlabel('Time (s)');ax.legend();save(fig,'06_predictive_target_explained.png')
        # Figure 7: signed world-frame position errors against posterior ±3 sigma.
        # This is a marginal (per-axis) uncertainty check, not a 2D radial bound.
        fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), dpi=150, sharex=True)
        for ax, axis in zip(axes, ('x', 'y')):
            signed_error = (values('ekf_' + axis) - values('true_' + axis)) * 1000
            bound = 3.0 * values('sigma_' + axis) * 1000
            ax.plot(t, signed_error, label=f'{axis.upper()} signed estimation error', lw=2.0)
            ax.plot(t, bound, '--', label='+3σ bound', lw=1.5)
            ax.plot(t, -bound, '--', label='−3σ bound', lw=1.5)
            ax.fill_between(t, -bound, bound, alpha=0.12, label='±3σ interval')
            ax.axhline(0, color='gray', alpha=0.5, lw=0.8)
            ax.axvline(MOVE_DURATION, ls=':', color='gray', label='Cube stops')
            ax.set_ylabel(f'{axis.upper()} error (mm)')
            ax.grid(alpha=0.18)
            ax.legend(loc='best', fontsize=9, ncol=3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('EKF signed position error and posterior ±3σ bounds', fontsize=18, weight='bold')
        save(fig, '07_EKF_uncertainty_3sigma.png')
        print('Saved results.csv + 7 cinematic 16:9 plots to:',OUT)
    except ImportError:
        print('Saved CSV to:',OUT,'(pip install matplotlib for plots)')

if __name__=='__main__':main()
