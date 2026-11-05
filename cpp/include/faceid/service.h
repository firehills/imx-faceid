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

/**
 * Route configured NATS subjects to inference, Qdrant, and FACE response codecs.
 * The service borrows its configuration/model/store; all three must remain
 * alive for the duration of run().
 */
class FaceIdService {
 public:
  /**
   * Bind this service to its dependencies; does not connect to NATS yet.
   * @param config Validated subjects, endpoint, and authentication settings.
   * @param model Initialized inference backend for image-based operations.
   * @param store Connected Qdrant store for enrolment and identity data.
   */
  FaceIdService(const AppConfig& config, EmbeddingModel& model,
                VectorStore& store);

  /**
   * Connect, register all handlers, and process messages until stop is requested.
   * @param stop_requested Signal-safe flag set by the process signal handler.
   * The call owns connection/subscription cleanup and propagates startup errors.
   */
  void run(const volatile std::sig_atomic_t& stop_requested);

 private:
  enum class Operation { Enrol, Query, UserInfo, DeleteUser, Debug };

  struct SubscriptionBinding {
    FaceIdService* service;          ///< Callback target; owner outlives subscription.
    Operation operation;             ///< Handler selected for this subject.
    natsSubscription* subscription;  ///< libnats subscription released at shutdown.
  };

  /** libnats callback: take ownership of message and dispatch via closure. */
  static void onMessage(natsConnection* connection,
                        natsSubscription* subscription, natsMsg* message,
                        void* closure);

  /** Create the configured NATS connection, applying the selected auth method. */
  void connect();
  /** Subscribe a subject and retain its callback binding until shutdown. */
  void subscribe(const std::string& subject, Operation operation);
  /** Route a message to the request parser/handler associated with its subject. */
  void dispatch(Operation operation, natsMsg* message);
  /** Parse enrolment, run inference, store raw/normalized vectors, and reply. */
  void handleEnrol(natsMsg* message);
  /** Parse a query, run inference, query Qdrant, and return candidate matches. */
  void handleQuery(natsMsg* message);
  /** Fetch and serialize a user's raw enrolment records. */
  void handleUserInfo(natsMsg* message);
  /** Delete one user's raw and normalized Qdrant records. */
  void handleDeleteUser(natsMsg* message);
  /** Collect and return Qdrant, inference, model, and NATS diagnostics. */
  void handleDebug(natsMsg* message);
  /** Publish serialized response bytes unless the request has no reply subject. */
  void publish(const std::string& reply_to, const std::vector<uint8_t>& payload);
  /** Log an exception with its operation and optional identity context. */
  void logFailure(const std::string& operation, const std::string& identity,
                  const std::exception& error) const;
  /** Return the shared per-user mutex used to serialize user mutations. */
  std::shared_ptr<std::mutex> userMutex(const std::string& user_id);
  /** Copy message payload bytes for verification and FlatBuffer decoding. */
  std::vector<uint8_t> messageData(const natsMsg* message) const;
  /** Drain when connected, then destroy this service's subscriptions/connection. */
  void shutdown();

  const AppConfig& config_;  ///< Borrowed settings; must outlive the service.
  EmbeddingModel& model_;    ///< Borrowed, mutex-serialized inference backend.
  VectorStore& store_;       ///< Borrowed persistent data interface.
  natsConnection* connection_;  ///< Owned libnats connection, nullable until connect.
  std::vector<std::unique_ptr<SubscriptionBinding> > subscriptions_;  ///< Owned bindings.
  std::mutex inference_mutex_;  ///< Serializes model calls across callback threads.
  std::mutex user_mutex_map_mutex_;  ///< Protects lazy per-user mutex creation.
  std::unordered_map<std::string, std::shared_ptr<std::mutex> > user_mutexes_;  ///< User locks.
};

}  // namespace faceid

#endif  // FACEID_SERVICE_H