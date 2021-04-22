(define (domain construction)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Element ?e)
    (Conf ?r ?q)
    (Grasp ?e ?g)

    (RemoveMovement ?r ?t ?e ?g)

    (Assembled ?e)
    (Removed ?e)
    (Order ?e1 ?e2)

    (Assigned ?r ?e)
  )

  ; remove an element from the assembled structure
  (:action remove_element
    ; :parameters (?r ?q1 ?t ?q2 ?e ?g)
    :parameters (?r ?t ?e ?g)
    :precondition (and
                       (RemoveMovement ?r ?t ?e ?g)
                       (Assembled ?e)
                       (forall (?e2) (imply (Order ?e ?e2) (Removed ?e2)))
                    ;    (HandEmpty ?r)
                    ;    (AtConf ?r ?q1)
                       )
    :effect (and (Removed ?e)
                 (not (Assembled ?e))
                ;  (not (HandEmpty ?r))
                ;  (Holding ?e)
                ;  (CanMove ?r)
                 )
  )
)

;   ; release an element to the rack
;   (:action release_element
;     :parameters (?r ?q ?e ?g)
;     :precondition (and
;                        (InitialConf ?r ?q)
;                     ;    (AtConf ?r ?q)
;                        (not (HandEmpty ?r))
;                        (Holding ?e)
;                        )
;     :effect (and (HandEmpty ?r)
;                  (CanMove ?r)
;                  (not (Holding ?e))
;                  )
;   )

;   (:action move_holding
;     :parameters (?r ?q1 ?t ?q2 ?e ?g)
;     :precondition (and
;                        (HoldingMovement ?r ?q1 ?t ?q2 ?e ?g)
;                        (AtConf ?r ?q1)
;                        (InitialConf ?r ?q2)
;                        (CanMove ?r)
;                        (Holding ?e)
;                        (not (HandEmpty ?r))
;                        )
;     :effect (and
;                  (not (AtConf ?r ?q1))
;                  (AtConf ?r ?q2)
;                  (not (CanMove ?r))
;                  )
;   )

;   (:action move_free
;     :parameters (?r ?q1 ?t ?q2)
;     :precondition (and
;                        (FreeMovement ?r ?q1 ?t ?q2)
;                        (AtConf ?r ?q1)
;                        (InitialConf ?r ?q1)
;                        (CanMove ?r)
;                        (HandEmpty ?r)
;                        )
;     :effect (and
;                  (not (AtConf ?r ?q1))
;                  (AtConf ?r ?q2)
;                  (not (CanMove ?r))
;                  )
;   )

; Connected handled within the stream function
;   (:derived (Connected ?e2)
;    (or (Grounded ?e2)
;        (exists (?e1) (and (Joined ?e1 ?e2)
;                           (Assembled ?e1)
;                           (Connected ?e1)
;                      )
;        )
;    )
;   )
