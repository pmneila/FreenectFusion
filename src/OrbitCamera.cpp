
#include "OrbitCamera.h"

#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>

#include "Eigen/Dense"

static void cross(double* res, const double* a, const double* b)
{
    res[0] = a[1]*b[2] - a[2]*b[1];
    res[1] = a[2]*b[0] - a[0]*b[2];
    res[2] = a[0]*b[1] - a[1]*b[0];
}

OrbitCamera::OrbitCamera()
    : mRadius(1000.0), mLongitude(3*M_PI/2.0),
    mLatitude(0.0)
{
    std::fill(mCenter, mCenter+3, 0.0);
    getVector();
}

OrbitCamera::~OrbitCamera() {}

void OrbitCamera::translate(double dx, double dy)
{
    double e1[3];
    double e2[3];
    static const double up[3] = {0.0, 1.0, 0.0};
    const double* v = getVector();
    cross(e1, v, up);
    cross(e2, v, e1);
    double c1 = dx/std::sqrt(std::inner_product(e1, e1+3, e1, 0.0));
    double c2 = dy/std::sqrt(std::inner_product(e2, e2+3, e2, 0.0));
    // Multiplies e1*c1 and e2*c2.
    std::transform(e1, e1+3, e1, std::bind1st(std::multiplies<double>(), c1));
    std::transform(e2, e2+3, e2, std::bind1st(std::multiplies<double>(), c2));
    // e1=e1+e2.
    std::transform(e1, e1+3, e2, e1, std::plus<double>());
    std::transform(e1, e1+3, mCenter, mCenter, std::plus<double>());
}

void OrbitCamera::rotate(double dlong, double dlat)
{
    mLongitude += dlong;
    mLatitude += dlat;
    if(mLatitude >= M_PI/2.0)
        mLatitude = M_PI/2.0 - 1e-3;
    else if(mLatitude <= -M_PI/2.0)
        mLatitude = -M_PI/2.0 + 1e-3;
}

void OrbitCamera::zoom(double dy)
{
    double step = mRadius/100.0;
    // ¿Por qué he hecho esto?
    if(dy <= -100.0)
        dy = -99.0;
    mRadius += step*dy;
    if(mRadius < 1.0)
        mRadius = 1.0;
}

const double* OrbitCamera::getPosition() const
{
    const double* vec = getVector();
    // position = center + vector.
    std::transform(mCenter, mCenter+3, vec, mPosition, std::plus<double>());
    return mPosition;
}

const double* OrbitCamera::getVector() const
{
    double cosLat = std::cos(mLatitude);
    mVector[0] = std::cos(mLongitude) * cosLat * mRadius;
    mVector[1] = std::sin(mLatitude) * mRadius;
    mVector[2] = std::sin(mLongitude) * cosLat * mRadius;
    return mVector;
}

#include <iostream>

const double* OrbitCamera::getTransform() const
{
    static const Eigen::Vector3d up(0.0, 1.0, 0.0);
    const double* aux = getPosition();
    Eigen::Vector3d pos(aux[0], aux[1], aux[2]);
    aux = getVector();
    Eigen::Vector3d z(aux[0], aux[1], aux[2]);
    z /= -z.norm();
    Eigen::Vector3d x = up.cross(z);
    x /= x.norm();
    Eigen::Vector3d y = z.cross(x);
    Eigen::Matrix3d rot;
    rot.block<1,3>(0,0) = x;
    rot.block<1,3>(1,0) = y;
    rot.block<1,3>(2,0) = z;
    
    typedef Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Matrix4dr;
    Matrix4dr transf = Matrix4dr::Identity();
    transf.block<3,3>(0,0) = rot.transpose();
    transf.block<3,1>(0,3) = pos;
    
    std::copy(transf.data(), transf.data()+16, mTransform);
    return mTransform;
}

void OrbitCamera::getGluLookAtParameters(double* params) const
{
    static const double up[3] = {0.0, 1.0, 0.0};
    const double* pos = getPosition();
    std::copy(pos, pos+3, params);
    std::copy(mCenter, mCenter+3, params+3);
    std::copy(up, up+3, params+6);
}
