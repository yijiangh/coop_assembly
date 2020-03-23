(define (stream kuka-tamp)
  ;;; input: object
  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  ;;; input: object, pose, grasp
  (:stream inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q ?t)
    :certified (and (Conf ?q)
                    (Traj ?t)
                    (Kin ?o ?p ?g ?q ?t)
               )
  )

  ;;; input: conf1, conf2
  (:stream plan-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :fluents (AtPose AtGrasp)
    :outputs (?t)
    :certified (IsMove ?q1 ?q2 ?t)
  )

  ;;; input: trajectory, object, pose
  (:predicate (TrajCollision ?t ?o2 ?p2)
    (and (Traj ?t)
         (Pose ?o2 ?p2)
    )
  )
)
