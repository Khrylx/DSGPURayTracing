#include "pathtracer.h"

#include <stack>
#include <random>
#include <algorithm>

#include "CMU462/CMU462.h"
#include "CMU462/vector3D.h"
#include "CMU462/matrix3x3.h"

#include "GL/glew.h"

#include "static_scene/brdf.h"
#include "static_scene/sphere.h"
#include "static_scene/triangle.h"
#include "static_scene/light.h"

using namespace CMU462::StaticScene;

namespace CMU462 {

//#define ENABLE_RAY_LOGGING 1

PathTracer::PathTracer(size_t max_ray_depth,
                       size_t ns_aa, size_t ns_area_light,
                       size_t ns_diff, size_t ns_glsy, size_t ns_refr,
                       size_t num_threads) {
  state = INIT,
  this->max_ray_depth = max_ray_depth;
  this->ns_aa = ns_aa;
  this->ns_area_light = ns_area_light;
  this->ns_diff = ns_diff;
  this->ns_glsy = ns_diff;
  this->ns_refr = ns_refr;

  bvh = NULL;
  scene = NULL;
  camera = NULL;

  gridSampler = new UniformGridSampler2D();
  hemisphereSampler = new UniformHemisphereSampler3D();

  show_rays = true;

  numWorkerThreads = num_threads;
  imageTileSize = 32;
  workerThreads.resize(numWorkerThreads);

}

PathTracer::~PathTracer() {

  delete bvh;
  delete gridSampler;
  delete hemisphereSampler;

}

void PathTracer::set_scene(Scene *scene) {

  if (state != INIT) {
    return;
  }

  if (this->scene != nullptr) {
    delete scene;
    delete bvh;
    selectionHistory.pop();
  }

  this->scene = scene;
  build_accel();

  if (has_valid_configuration()) {
    state = READY;
  }
}

void PathTracer::set_camera(Camera *camera) {

  if (state != INIT) {
    return;
  }

  this->camera = camera;
  if (has_valid_configuration()) {
    state = READY;
  }

}

void PathTracer::set_frame_size(size_t width, size_t height) {
  if (state != INIT && state != READY) {
    stop();
  }
  frameBuffer.resize(width, height);
  if (has_valid_configuration()) {
    state = READY;
  }
}

bool PathTracer::has_valid_configuration() {
  return scene && camera && gridSampler && hemisphereSampler &&
         (!frameBuffer.is_empty());
}

void PathTracer::update_screen() {
  switch (state) {
    case INIT:
    case READY:
      break;
    case VISUALIZE:
      visualize_accel();
      break;
    case RENDERING:
    case DONE:
      glDrawPixels(frameBuffer.width(), frameBuffer.height(), GL_RGBA,
                   GL_UNSIGNED_BYTE, frameBuffer.pixels());
      break;
  }
}

void PathTracer::stop() {
  switch (state) {
    case INIT:
    case READY:
      break;
    case VISUALIZE:
      while (selectionHistory.size() > 1) {
        selectionHistory.pop();
      }
      state = READY;
      break;
    case RENDERING:
      continueRaytracing = false;
    case DONE:
        for (int i=0; i<numWorkerThreads; i++) {
            workerThreads[i]->join();
            delete workerThreads[i];
        }
        frameBuffer.clear();
        state = READY;
      break;
  }
}

void PathTracer::clear() {
  if (state != READY) return;
  delete bvh;
  bvh = NULL;
  scene = NULL;
  camera = NULL;
  selectionHistory.pop();
  frameBuffer.resize(0, 0);
  state = INIT;
}

void PathTracer::start_visualizing() {
  if (state != READY) {
    return;
  }
  state = VISUALIZE;
}

void PathTracer::start_raytracing() {
  if (state != READY) {
    return;
  }

  rayLog.clear();
  workQueue.clear();

  state = RENDERING;
  continueRaytracing = true;
  workerDoneCount = 0;

  // populate the tile work queue
  if (numWorkerThreads == 1) {
      workQueue.put_work(
        WorkItem(0, 0, frameBuffer.width(), frameBuffer.height())
      );
  } else {
      for (size_t y = 0; y < frameBuffer.height(); y += imageTileSize) {
          for (size_t x = 0; x < frameBuffer.width(); x += imageTileSize) {
              workQueue.put_work(WorkItem(x, y, imageTileSize, imageTileSize));
          }
      }
  }

  // launch threads
  for (int i=0; i<numWorkerThreads; i++) {
      workerThreads[i] = new std::thread(&PathTracer::worker_thread, this);
  }
}


void PathTracer::build_accel() {

  // collecte primitives //
  fprintf(stdout, "[PathTracer] Collecting primitives... ");
  timer.start();
  vector<Primitive *> primitives;
  for (SceneObject *obj : scene->objects) {
    const vector<Primitive *> &obj_prims = obj->get_primitives();
    primitives.reserve(primitives.size() + obj_prims.size());
    primitives.insert(primitives.end(), obj_prims.begin(), obj_prims.end());
  }
  timer.stop();
  fprintf(stdout, "Done! (%.4f sec)\n", timer.duration());

  // build BVH //
  fprintf(stdout, "[PathTracer] Building BVH... ");
  timer.start();
  bvh = new BVHAccel(primitives);
  timer.stop();
  fprintf(stdout, "Done! (%.4f sec)\n", timer.duration());

  // initial visualization //
  selectionHistory.push(bvh->get_root());
}

void PathTracer::log_ray_miss(const Ray& r) {
    rayLog.push_back(LoggedRay(r, -1.0));
}

void PathTracer::log_ray_hit(const Ray& r, double hit_t) {
    rayLog.push_back(LoggedRay(r, hit_t));
}

void PathTracer::visualize_accel() const {

  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);
  glLineWidth(1);
  glEnable(GL_DEPTH_TEST);

  // hardcoded color settings
  Color cnode = Color(.5, .5, .5, .25);
  Color cnode_hl = Color(1., .25, .0, .6);
  Color cnode_hl_child = Color(1., 1., 1., .6);

  //Color cprim = Color(.3, .3, .3, .5);
  Color cprim_hl_left = Color(.6, .6, 1., 1);
  Color cprim_hl_right = Color(.8, .8, 1., 1);
  Color cprim_hl_edges = Color(0., 0., 0., 0.5);

  BVHNode *selected = selectionHistory.top();

  // render solid geometry (with depth offset)
  glPolygonOffset(1.0, 1.0);
  glEnable(GL_POLYGON_OFFSET_FILL);

  if (selected->isLeaf()) {
    for (size_t i = 0; i < selected->range; ++i) {
       bvh->primitives[selected->start + i]->draw(cprim_hl_left);
    }
  } else {
      if (selected->l) {
          BVHNode* child = selected->l;
          for (size_t i = 0; i < child->range; ++i) {
              bvh->primitives[child->start + i]->draw(cprim_hl_left);
          }
      }
      if (selected->r) {
          BVHNode* child = selected->r;
          for (size_t i = 0; i < child->range; ++i) {
              bvh->primitives[child->start + i]->draw(cprim_hl_right);
          }
      }
  }

  glDisable(GL_POLYGON_OFFSET_FILL);

  // draw geometry outline
  for (size_t i = 0; i < selected->range; ++i) {
      bvh->primitives[selected->start + i]->drawOutline(cprim_hl_edges);
  }

  // keep depth buffer check enabled so that mesh occluded bboxes, but
  // disable depth write so that bboxes don't occlude each other.
  glDepthMask(GL_FALSE);

  // create traversal stack
  stack<BVHNode *> tstack;

  // push initial traversal data
  tstack.push(bvh->get_root());

  // draw all BVH bboxes with non-highlighted color
  while (!tstack.empty()) {

    BVHNode *current = tstack.top();
    tstack.pop();

    current->bb.draw(cnode);
    if (current->l) tstack.push(current->l);
    if (current->r) tstack.push(current->r);
  }

  // draw selected node bbox and primitives
  if (selected->l) selected->l->bb.draw(cnode_hl_child);
  if (selected->r) selected->r->bb.draw(cnode_hl_child);

  glLineWidth(3.f);
  selected->bb.draw(cnode_hl);

  // now perform visualization of the rays
  if (show_rays) {
      glLineWidth(1.f);
      glBegin(GL_LINES);

      for (size_t i=0; i<rayLog.size(); i+=500) {

          const static double VERY_LONG = 10e4;
          double ray_t = VERY_LONG;

          // color rays that are hits yellow
          // and rays this miss all geometry red
          if (rayLog[i].hit_t >= 0.0) {
              ray_t = rayLog[i].hit_t;
              glColor4f(1.f, 1.f, 0.f, 0.1f);
          } else {
              glColor4f(1.f, 0.f, 0.f, 0.1f);
          }

          Vector3D end = rayLog[i].o + ray_t * rayLog[i].d;

          glVertex3f(rayLog[i].o[0], rayLog[i].o[1], rayLog[i].o[2]);
          glVertex3f(end[0], end[1], end[2]);
      }
      glEnd();
  }

  glDepthMask(GL_TRUE);
  glPopAttrib();
}

void PathTracer::key_press(int key) {
  if (state != VISUALIZE) {
    return;
  }
  BVHNode *current = selectionHistory.top();
  switch (key) {
    case KEYBOARD_UP:
      if (current != bvh->get_root()) {
        selectionHistory.pop();
      }
      break;
    case KEYBOARD_LEFT:
      if (current->l) {
        selectionHistory.push(current->l);
      }
      break;
    case KEYBOARD_RIGHT:
      if (current->l) {
        selectionHistory.push(current->r);
      }
      break;
  case 'a':
  case 'A':
      show_rays = !show_rays;
    default:
      return;
  }
}

Spectrum PathTracer::trace_ray(const Ray &r) {

  Intersection isect;

  if (!bvh->intersect(r, &isect)) {

    // log ray miss
    #ifdef ENABLE_RAY_LOGGING
    log_ray_miss(r);
    #endif

    // black environment map
    return Spectrum(0,0,0);
  }

  // log ray hit
  #ifdef ENABLE_RAY_LOGGING
  log_ray_hit(r, isect.t);
  #endif

    //return Spectrum(1,0,0);
    
  Spectrum L_out(0,0,0);

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D hit_n = isect.n;

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();
    
    
  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  Vector3D w_out = w2o * (r.o - hit_p);
  w_out.normalize();

  // Note to students: while testing part 1 of the assignment, you can
  // choose to use either a hemispherical light source or a simple
  // directional light.  The directional light might be easier for
  // quick debugging, but the hemispherical light gives much more
  // pleasing results.

  InfiniteHemisphereLight light(Spectrum(1.f, 1.f, 1.f));
  //DirectionalLight light(Spectrum(5.f, 5.f, 5.f), Vector3D(1.0, 0.0, 0.0));

  Vector3D dir_to_light;
  double dist_to_light;
  double pdf;

  // no need to take multiple samples from a directional source
  int num_light_samples = light.is_delta_light() ? 1 : ns_area_light;

  // integrate light over the hemisphere about the normal
  double scale = 1.0 / num_light_samples;
  for (int i=0; i<num_light_samples; i++) {

      // returns a vector 'dir_to_light' that is a direction from
      // point hit_p to the point on the light source.  It also returns
      // the distance from point x to this point on the light source.
      // (pdf is the probability of randomly selecting the random
      // sample point on the light source -- more on this in part 2)
      Spectrum light_L = light.sample_L(hit_p, &dir_to_light, &dist_to_light, &pdf);

      // convert direction into coordinate space of the surface, where
      // the surface normal is [0 0 1]
      //dir_to_light = Vector3D(0,0,1);
      Vector3D w_in = w2o * dir_to_light;

      // note that computing dot(n,w_in) is simple
      // in surface coordinates since the normal is [0 0 1]
      double cos_theta = std::max(0.0, w_in[2]);

      // evaluate surface brdf
      Spectrum f = isect.brdf->f(w_out, w_in);

      
      light_L.r *= f.r;
      light_L.g *= f.g;
      light_L.b *= f.b;
      //cout << w_in <<endl;
      L_out += cos_theta/pdf*light_L;
      
      // TODO:
      // construct a shadow ray and compute whether the intersected surface is
      // in shadow and accumulate reflected radiance
  }
    //cout << L_out*scale <<endl;

  return L_out*scale;
}

void PathTracer::raytrace_pixel(size_t x, size_t y) {

  // TODO:
  // Sample the pixel with coordinate (x,y) and update the frame buffer
  // accordingly. The sample rate is given by the number of camera rays
  // per pixel.

    size_t w = frameBuffer.width();
    size_t h = frameBuffer.height();
    
  int num_samples = ns_aa;

//  Vector2D p = Vector2D(0.5,0.5);
//  Spectrum s = trace_ray(camera->generate_ray(p.x, p.y));
//  frameBuffer.put_color(s.toColor(), x, y);
    
    Spectrum s(0,0,0);
    for (int i = 0; i < num_samples; i++) {
        Vector2D rp = gridSampler->get_sample();
        double px = (x+rp.x) / w;
        double py = (y+rp.y) / h;
        Ray r = camera->generate_ray(px, py);
        s += trace_ray(r);
    }
    s *= 1.0/num_samples;
    
    frameBuffer.put_color(s.toColor(), x, y);
    

    
}

void PathTracer::raytrace_tile(int tile_x, int tile_y, int tile_w, int tile_h) {

    size_t w = frameBuffer.width();
    size_t h = frameBuffer.height();

    size_t tile_start_x = tile_x;
    size_t tile_start_y = tile_y;

    size_t tile_end_x = std::min(tile_start_x + tile_w, w);
    size_t tile_end_y = std::min(tile_start_y + tile_h, h);

    for (size_t y = tile_start_y; y < tile_end_y; y++) {
        for (size_t x = tile_start_x; x < tile_end_x; x++) {
            raytrace_pixel(x, y);
        }
        if (!this->continueRaytracing) {
            fprintf(stdout, "[PathTracer thread] Rendering canceled!\n");
            return;
        }
    }
}

void PathTracer::worker_thread() {

    Timer timer;
    timer.start();

    while (continueRaytracing && !workQueue.is_empty()) {
        WorkItem work = workQueue.get_work();
        raytrace_tile(work.tile_x, work.tile_y, work.tile_w, work.tile_h);
    }

    workerDoneCount++;

    if (continueRaytracing && workerDoneCount == numWorkerThreads) {
        timer.stop();
        fprintf(stdout, "[PathTracer] Rendering completed! (%.4fs)\n", timer.duration());
        state = DONE;
    }
}

void PathTracer::increase_area_light_sample_count() {
    ns_area_light *= 2;
    fprintf(stdout, "[PathTracer] Area light sample count increased to %zu!\n", ns_area_light);
}

void PathTracer::decrease_area_light_sample_count() {
    if (ns_area_light > 1)
        ns_area_light /= 2;
    fprintf(stdout, "[PathTracer] Area light sample count decreased to %zu!\n", ns_area_light);
}


}  // namespace CMU462
