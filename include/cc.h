#pragma once

#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"
#include <random>
#include <cmath>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Joy.h>


#include <torch/script.h> // Include the PyTorch C++ API
#include <queue>


template <class T>
class limited_queue{
    public:

    limited_queue(int maxlen_) : maxlen(maxlen_){}

    void push(T input){
        if (my_queue.size() + 1 > maxlen){
            my_queue.pop();
        }
        my_queue.push(input);
    }

    T pop(){
        T ret = my_queue.front();
        my_queue.pop();
        return ret;
    }

    size_t size() const {return my_queue.size();}

    bool is_empty() const {return my_queue.empty();}

    std::queue<T> data() const {return my_queue;}

    private:
        size_t maxlen;
        std::queue<T> my_queue;
};


inline torch::Tensor queue_to_tensor(limited_queue<torch::Tensor> tensorQueue){
    // Check if the queue is empty
    if (tensorQueue.is_empty()) {
        // Return an empty tensor if the queue is empty
        throw std::invalid_argument("Queue cannot be empty");
    }

    // Get the first tensor from the queue

    torch::Tensor result = tensorQueue.pop(); // Remove the first tensor from the queue
    
    // Iterate over the remaining tensors in the queue
    while (!tensorQueue.is_empty()) {
        // Get the next tensor from the queue
        torch::Tensor nextTensor = tensorQueue.pop(); // Remove the tensor from the queue

        // Concatenate the next tensor along the specified dimension
        result = torch::cat({result, nextTensor}, 1);
        // result = torch::cat({nextTensor, result}, 1);
    }
    return result;
}

template <class T>
inline std::vector<T> queue_to_vector(const std::queue<T>& q) {
    std::vector<T> result;
    std::queue<T> tempQueue = q; // Create a copy of the input queue

    // Iterate through the elements of the queue and push them into the vector
    while (!tempQueue.empty()) {
        result.push_back(tempQueue.front());
        tempQueue.pop();
    }

    return result;
}

template <class T>
inline torch::Tensor Eigen_to_tensor(const T& vec){
    torch::Tensor ret = torch::empty(vec.size());
    for (int i = 0; i < vec.size(); i++){
        ret[i] = vec[i];
    }
    return ret;
}

template <class T>
inline T tensor_to_Eigen(const torch::Tensor& tensor){
    int dim = tensor.sizes()[0];
    T ret;
    for (int i = 0; i < dim; i++){
        ret[i] = static_cast<double>(tensor[i].item<float>());
    }
    return ret;
}

Eigen::Vector3d quat_rotate_inverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) 
{
    Eigen::Vector3d q_vec = q.vec();
    double q_w = q.w();

    Eigen::Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Eigen::Vector3d b = q_vec.cross(v) * q_w * 2.0;
    Eigen::Vector3d c = q_vec * (q_vec.dot(v) * 2.0);

    return a - b + c;
}

//TYPEDEF
typedef Eigen::Matrix<float, 3, 1> Vector3f;


class CustomController
{
public:
    CustomController(RobotData &rd);
    Eigen::VectorQd getControl();

    //void taskCommandToCC(TaskCommand tc_);
    
    void computeSlow();
    void computeFast();
    void computePlanner();
    void copyRobotData(RobotData &rd_l);

    RobotData &rd_;
    RobotData rd_cc_;

    //////////////////////////////////////////// Donghyeon RL /////////////////////////////////////////
    void loadNetwork();
    void processNoise();
    void processObservation();
    void feedforwardPolicy();
    void initVariable();
    Eigen::Vector3d mat2euler(Eigen::Matrix3d mat);

    static const int num_action = 12; // Actuated actions + anything else (maybe phase?)
    static const int num_actuator_action = 12;
    static const int num_cur_state = 50;
    static const int num_cur_internal_state = 37;
    static const int num_state_skip = 2;
    static const int num_state_hist = 5;
    static const int num_state = num_cur_internal_state*num_state_hist+num_action*(num_state_hist-1);
    static const int num_hidden = 256;

    Eigen::MatrixXd policy_net_w0_;
    Eigen::MatrixXd policy_net_b0_;
    Eigen::MatrixXd policy_net_w2_;
    Eigen::MatrixXd policy_net_b2_;
    Eigen::MatrixXd action_net_w_;
    Eigen::MatrixXd action_net_b_;
    Eigen::MatrixXd hidden_layer1_;
    Eigen::MatrixXd hidden_layer2_;
    Eigen::MatrixXd rl_action_;

    Eigen::MatrixXd value_net_w0_;
    Eigen::MatrixXd value_net_b0_;
    Eigen::MatrixXd value_net_w2_;
    Eigen::MatrixXd value_net_b2_;
    Eigen::MatrixXd value_net_w_;
    Eigen::MatrixXd value_net_b_;
    Eigen::MatrixXd value_hidden_layer1_;
    Eigen::MatrixXd value_hidden_layer2_;
    double value_;

    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;
    
    Eigen::MatrixXd state_;
    Eigen::MatrixXd state_cur_;
    Eigen::MatrixXd state_buffer_;
    Eigen::MatrixXd state_mean_;
    Eigen::MatrixXd state_var_;

    std::ofstream writeFile;

    float phase_ = 0.0;

    bool is_on_robot_ = false;
    bool is_write_file_ = true;
    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_rl_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;

    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    float start_time_;
    float time_inference_pre_ = 0.0;
    float time_write_pre_ = 0.0;

    double time_cur_;
    double time_pre_;
    double action_dt_accumulate_ = 0.0;

    Eigen::Vector3d euler_angle_;

    // float ft_left_init_ = 500.0;
    // float ft_right_init_ = 500.0;

    // Joystick
    ros::NodeHandle nh_;

    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    ros::Subscriber joy_sub_;

    double target_vel_x_ = 0.0;
    double target_vel_y_ = 0.0;

    //////////////////////////////////////////// Woohyun RL /////////////////////////////////////////
    
    torch::jit::script::Module module;

    torch::Tensor observation;
    torch::Tensor action;

    int history_length = 15;
    int observation_size = 44;

    Vector3d commands = Vector3d(0., 0., 0.);
    limited_queue<torch::Tensor> observation_history = limited_queue<torch::Tensor>(history_length);
    Vector3d gravity = Vector3d(0, 0, -GRAVITY).normalized();


    // Scales
    double lin_vel_scale = 1.0;
    double ang_vel_scale = 0.25;
    double dof_pos_scale = 1.0;
    double dof_vel_scale = 0.05;
    // height_measurements_scale = 5.0
    // ext_forces_scale = 0.1
    // ext_torques_scale = 1.
    // friction_coeffs_scale = 1.
    // dof_friction_scale = 10.
    // dof_damping_scale = 10.
    float clip_observations = 100.;
    float clip_actions = 5.;
    float action_scale = 100.;

private:
    Eigen::VectorQd ControlVal_;
};