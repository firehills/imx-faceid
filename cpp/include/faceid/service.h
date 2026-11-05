#ifndef FACEID_SERVICE_H
#define FACEID_SERVICE_H

#include <csignal>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nats/nats.h>

#include "faceid/config.h"
#include "faceid/inference.h"
#include "faceid/vector_store.h"

namespace faceid {

/** Subscribe to the configured NATS subjects and handle FACE FlatBuffer requests. */
class FaceIdService {
 public:
  /** Bind the service to its validated configuration, model, and Qdrant store. */
  FaceIdService(const AppConfig& config, EmbeddingModel& model,
                VectorStore& store);

  /** Connect, subscribe, and serve until the signal-safe stop flag is set. */
  void run(const volatile std::sig_atomic_t& stop_requested);

 private:
  enum class Operation { Enrol, Query, UserInfo, DeleteUser, Debug };

  struct SubscriptionBinding {
    FaceIdService* service;
    Operation operation;
    natsSubscription* subscription;
  };

  static void onMessage(natsConnection* connection,
                        natsSubscription* subscription, natsMsg* message,
                        void* closure);

  void connect();
  void subscribe(const std::string& subject, Operation operation);
  void dispatch(Operation operation, natsMsg* message);
  void handleEnrol(natsMsg* message);
  void handleQuery(natsMsg* message);
  void handleUserInfo(natsMsg* message);
  void handleDeleteUser(natsMsg* message);
  void handleDebug(natsMsg* message);
  void publish(const std::string& reply_to, const std::vector<uint8_t>& payload);
  void logFailure(const std::string& operation, const std::string& identity,
                  const std::exception& error) const;
  std::shared_ptr<std::mutex> userMutex(const std::string& user_id);
  std::vector<uint8_t> messageData(const natsMsg* message) const;
  void shutdown();

  const AppConfig& config_;
  EmbeddingModel& model_;
  VectorStore& store_;
  natsConnection* connection_;
  std::vector<std::unique_ptr<SubscriptionBinding> > subscriptions_;
  std::mutex inference_mutex_;
  std::mutex user_mutex_map_mutex_;
  std::unordered_map<std::string, std::shared_ptr<std::mutex> > user_mutexes_;
};

}  // namespace faceid

#endif  // FACEID_SERVICE_H