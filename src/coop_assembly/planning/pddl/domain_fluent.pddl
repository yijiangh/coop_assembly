(define (domain construction)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Element ?e)
    (Conf ?r ?q)
    (InitialConf ?r ?q)
    (Grasp ?e ?g)
    ; (RemoveStartConf ?r ?q)
    ; (RemoveEndConf ?r ?q)

    (RemoveMovement ?r ?q1 ?t ?q2 ?e ?g)
    (HoldingMovement ?r ?q1 ?t2 ?q2 ?e ?g)
    (FreeMovement ?r ?q1 ?t2 ?q2)

    (Assembled ?e)
    (Removed ?e)
    (Grounded ?e)
    (Connected ?e)
    (Joined ?e1 ?e2)
    (Order ?e1 ?e2)

    (CanMove ?r)
    (HandEmpty ?r)
    (AtConf ?r ?q)
    (Assigned ?r ?e)
    ; (Stiff)
  )

  ; remove an element from the assembled structure
  (:action remove_element
    :parameters (?r ?q1 ?t ?q2 ?e ?g)
    :precondition (and
                       (RemoveMovement ?r ?q1 ?t ?q2 ?e ?g)
                       (Assembled ?e)
                       ; (Stiff)
                       ; (Connected ?e)
                       )
    :effect (and (Removed ?e)
                 (not (Assembled ?e))
                 (not (HandEmpty ?r))
                 (CanMove ?r)
                 )
  )

  ; release an element to the rack
  (:action release_element
    :parameters (?r ?q ?e ?g)
    :precondition (and
                       (InitialConf ?r ?q)
                       (AtConf ?r ?q)
                       (Removed ?e)
                       )
    :effect (and (Removed ?e)
                 (not (Assembled ?e))
                 (HandEmpty ?r)
                 (CanMove ?r)
                 )
  )

  (:action move_holding
    :parameters (?r ?q1 ?t ?q2 ?e ?g)
    :precondition (and
                       (HoldingMovement ?r ?q1 ?t ?q2 ?e ?g)
                       (AtConf ?r ?q1)
                       (InitialConf ?r ?q2)
                       (CanMove ?r)
                       )
    :effect (and
                 (not (AtConf ?r ?q1))
                 (AtConf ?r ?q2)
                 (not (CanMove ?r))
                 )
  )

  (:action move_free
    :parameters (?r ?q1 ?t ?q2)
    :precondition (and
                       (FreeMovement ?r ?q1 ?t ?q2)
                       (AtConf ?r ?q1)
                       (InitialConf ?r ?q1)
                       (CanMove ?r)
                       (HandEmpty ?r)
                       )
    :effect (and
                 (not (AtConf ?r ?q1))
                 (AtConf ?r ?q2)
                 (not (CanMove ?r))
                 )
  )

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
)
